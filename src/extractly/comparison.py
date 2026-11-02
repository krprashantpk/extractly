"""Declare evaluated Pydantic fields and caller-owned comparison functions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from inspect import Parameter, iscoroutinefunction, signature
from types import UnionType
from typing import Annotated, ClassVar, ParamSpec, Union, cast, get_args, get_origin

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainValidator,
    TypeAdapter,
    ValidationError,
    WrapValidator,
)

Parameters = ParamSpec("Parameters")


type Comparator = Callable[[object, object], bool]


def field_comparator(*field_names: str) -> Callable[[Callable[Parameters, bool]], Callable[Parameters, bool]]:
    """Register a synchronous static comparator for one or more model fields.

    Place this decorator below staticmethod and directly above the function.
    Methods may have arbitrary names and must accept expected and actual values
    positionally and return bool. Registration does not wrap the callable.
    """
    if any(not isinstance(name, str) for name in field_names):
        raise TypeError("Comparator field names must be strings")
    if not field_names or any(not name.strip() for name in field_names):
        raise ValueError("A comparator requires nonempty field names")
    if len(set(field_names)) != len(field_names):
        raise ValueError("Comparator field names must be unique")

    def register(function: Callable[Parameters, bool]) -> Callable[Parameters, bool]:
        """Attach field bindings without changing the function signature."""
        if not callable(function) or isinstance(function, (staticmethod, classmethod)):
            raise TypeError("Place field_comparator directly above a comparison function, below staticmethod")
        if hasattr(function, "__comparison_fields__"):
            raise ValueError("Use one field_comparator listing all fields instead of stacked decorators")
        function.__dict__["__comparison_fields__"] = field_names
        return function

    return register


def equal_values(expected: object, actual: object) -> bool:
    """Compare normalized values without treating booleans as integers."""
    return expected == actual and isinstance(expected, bool) == isinstance(actual, bool)


class EvaluationModel(BaseModel):
    """Declare fields evaluated within each canonical segment's fields object.

    Undeclared CU fields remain available for alignment and context but are not
    scored. Nested evaluation models and lists describe nested occurrences.
    Pydantic types, aliases, requiredness, and Field constraints define input
    validation; custom transformation validators are not supported. Defaults
    describe optionality and never manufacture extracted values for scoring.
    """

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    comparison_version: ClassVar[str] = "1"
    _comparators: ClassVar[dict[str, Comparator]] = {}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: object) -> None:
        """Bind comparators once and reject unsupported or ambiguous declarations."""
        super().__pydantic_init_subclass__(**kwargs)
        decorators = cls.__pydantic_decorators__
        if decorators.validators or decorators.field_validators or decorators.root_validators or decorators.model_validators:
            raise ValueError("EvaluationModel supports Field constraints, not custom Pydantic validators")
        if not isinstance(cls.comparison_version, str) or not cls.comparison_version.strip():
            raise ValueError("comparison_version must be a nonempty string")
        comparators: dict[str, Comparator] = dict.fromkeys(cls.model_fields, equal_values)
        effective_members = {name: member for base in reversed(cls.__mro__) for name, member in vars(base).items()}
        for base in reversed(cls.__mro__):
            registered: set[str] = set()
            for name, member in vars(base).items():
                if effective_members[name] is not member:
                    continue
                function = member.__func__ if isinstance(member, (staticmethod, classmethod)) else member
                field_names = cast(tuple[str, ...], getattr(function, "__comparison_fields__", ()))
                if not field_names:
                    continue
                if not isinstance(member, staticmethod) or iscoroutinefunction(function):
                    raise TypeError("Field comparators must be synchronous static methods")
                parameters = tuple(signature(cast(Comparator, function)).parameters.values())
                if len(parameters) != 2 or any(parameter.kind not in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD) for parameter in parameters):
                    raise TypeError("Field comparators must accept exactly expected and actual positionally")
                for field_name in field_names:
                    if field_name not in cls.model_fields:
                        raise ValueError(f"Unknown comparator field: {field_name}")
                    if field_name in registered:
                        raise ValueError(f"Multiple comparators registered for field: {field_name}")
                    registered.add(field_name)
                    comparators[field_name] = cast(Comparator, function)
        cls._comparators = comparators


@dataclass(frozen=True)
class ModelField:
    """Bind an occurrence pattern to its Pydantic validator and comparison policy."""

    validator: TypeAdapter[object]
    comparator: Comparator
    owner: type[EvaluationModel]

    def compare(self, expected: object, actual: object) -> bool:
        """Invoke the registered function and validate its result without catching caller errors."""
        result = self.comparator(expected, actual)
        if type(result) is not bool:
            raise TypeError("Field comparators must return bool")
        return result


@dataclass(frozen=True)
class ModelContract:
    """Compile one model into reusable validators, selectors, and policy metadata.

    The model applies to each canonical segment's fields object. Values remain
    in their original documents for row matching, presence checks, and evidence.
    Only present leaf values are validated and passed to comparison functions.
    """

    model: type[EvaluationModel]
    fields: dict[str, ModelField]
    schema: dict[str, object]

    @classmethod
    def compile(cls, model: type[EvaluationModel]) -> ModelContract:
        """Resolve nested fields and aliases, rejecting unsupported declarations early."""
        fields: dict[str, ModelField] = {}

        def walk(annotation: object, path: str, owner: type[EvaluationModel], comparator: Comparator, ancestors: tuple[type[EvaluationModel], ...]) -> None:
            """Compile arrays and nested models without expanding source occurrences."""
            base = annotation
            while get_origin(base) is Annotated:
                arguments = get_args(base)
                if any(isinstance(metadata, (BeforeValidator, AfterValidator, PlainValidator, WrapValidator)) for metadata in arguments[1:]):
                    raise ValueError("EvaluationModel does not support transforming Annotated validators")
                base = arguments[0]
            if get_origin(base) in (Union, UnionType):
                members = [member for member in get_args(base) if member is not type(None)]
                if len(members) == 1:
                    base = members[0]
                elif any(get_origin(member) is list or (isinstance(member, type) and issubclass(member, BaseModel)) for member in members):
                    raise ValueError("Evaluation models cannot contain ambiguous unions of containers")
            if comparator is equal_values and get_origin(base) is list:
                walk(get_args(base)[0], f"{path}/*", owner, equal_values, ancestors)
            elif comparator is equal_values and isinstance(base, type) and issubclass(base, BaseModel):
                if not issubclass(base, EvaluationModel):
                    raise TypeError("Nested evaluation models must inherit EvaluationModel")
                if base in ancestors:
                    raise ValueError("Recursive evaluation models are not supported")
                if not base.model_fields:
                    raise ValueError("Evaluation models must declare at least one field")
                aliases: set[str] = set()
                for name, field in base.model_fields.items():
                    alias = field.validation_alias if field.validation_alias is not None else field.alias if field.alias is not None else name
                    if not isinstance(alias, str) or not alias or alias == "*":
                        raise ValueError("Evaluation fields require simple nonempty aliases without wildcard names")
                    if alias in aliases:
                        raise ValueError("Evaluation fields must have distinct input aliases")
                    aliases.add(alias)
                    child = f"{path}/{alias.replace('~', '~0').replace('/', '~1')}"
                    walk(field.rebuild_annotation(), child, base, base._comparators[name], (*ancestors, base))
            else:
                if any(get_origin(argument) is Annotated for argument in get_args(base)):
                    raise ValueError("Annotated union members require an explicit supported field type")
                adapter: TypeAdapter[object] = TypeAdapter(cast(type[object], annotation), config=None if isinstance(base, type) and issubclass(base, BaseModel) else owner.model_config)
                fields[path] = ModelField(adapter, comparator, owner)

        walk(model, "/segments/*/fields", model, equal_values, ())
        return cls(model, fields, model.model_json_schema(mode="validation"))

    def validate_gold(self, value: object) -> None:
        """Reject malformed canonical gold or invalid model values before comparison."""
        if not isinstance(value, dict) or not isinstance(value.get("segments"), list) or not value["segments"]:
            raise ValueError("Model evaluation requires gold with a nonempty segments array")
        for segment in value["segments"]:
            if not isinstance(segment, dict) or "fields" not in segment:
                raise ValueError("Every gold segment requires a fields object")
            self.model.model_validate(segment["fields"])

    def violation_paths(self, value: object) -> list[str]:
        """Return canonical prediction paths that violate the Pydantic contract."""
        if not isinstance(value, dict) or not isinstance(value.get("segments"), list) or not value["segments"]:
            return ["/segments"]
        paths: list[str] = []
        for index, segment in enumerate(value["segments"]):
            prefix = f"/segments/{index}/fields"
            if not isinstance(segment, dict) or "fields" not in segment:
                paths.append(prefix)
                continue
            try:
                self.model.model_validate(segment["fields"])
            except ValidationError as exc:
                paths.extend(prefix + "".join(f"/{str(part).replace('~', '~0').replace('/', '~1')}" for part in error["loc"]) for error in exc.errors())
        return sorted(set(paths))

    def describe(self) -> dict[str, object]:
        """Serialize an auditable contract without importing executable code from JSON.

        comparison_version is caller-owned and must change when a comparator's
        policy changes. JSON snapshots describe policy but cannot reconstruct
        executable model classes; Python-mode config copies retain those classes.
        """
        return {
            "model": f"{self.model.__module__}.{self.model.__qualname__}",
            "schema": self.schema,
            "comparators": {path: {
                "callable": f"{field.comparator.__module__}.{field.comparator.__qualname__}",
                "version": field.owner.comparison_version,
            } for path, field in self.fields.items()},
        }