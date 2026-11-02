"""Preserve CU analyzer rules and validate supplied values without inventing fields."""

import copy
import json
import re

from jsonschema import Draft202012Validator, FormatChecker

from ..types import JsonObject
from .cu_result_parser import pointer

VALUE_KEYS = {"string": "valueString", "date": "valueDate", "time": "valueTime", "number": "valueNumber", "integer": "valueInteger", "boolean": "valueBoolean", "json": "valueJson"}


def value_schema(definition: JsonObject) -> JsonObject:
    """Translate CU value types for closed-schema checks, retaining optionality."""
    if not isinstance(definition, dict) or definition.get("type") not in (*VALUE_KEYS, "array", "object"):
        raise ValueError("Every analyzer field must declare a supported CU type")
    kind = definition["type"]
    schema: JsonObject = {} if kind == "json" else {"type": "string" if kind in ("date", "time") else kind}
    if kind == "date":
        schema["format"] = "date"
    elif kind == "time":
        schema["pattern"] = r"^(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\.[0-9]+)?$"
    elif kind == "array":
        schema["items"] = value_schema(definition.get("items", {}))
    elif kind == "object":
        properties = definition.get("properties")
        if not isinstance(properties, dict):
            raise ValueError("Analyzer object fields must declare properties")
        schema.update(properties={name: value_schema(child) for name, child in properties.items()}, additionalProperties=False)
    if "enum" in definition:
        schema["enum"] = copy.deepcopy(definition["enum"])
    return schema


def analyzer_schema(definition: JsonObject) -> tuple[JsonObject, Draft202012Validator]:
    """Accept a full analyzer definition or fieldSchema without losing its original rules."""
    if not isinstance(definition, dict):
        raise TypeError("analyzer_definition must be a CU analyzer or fieldSchema dictionary")
    schema = copy.deepcopy(definition.get("fieldSchema", definition))
    if not isinstance(schema, dict) or not isinstance(schema.get("fields"), dict) or not schema["fields"]:
        raise ValueError("analyzer_definition must contain a nonempty fieldSchema.fields mapping")
    json.dumps(schema, allow_nan=False)
    converted = {"type": "object", "properties": {name: value_schema(child) for name, child in schema["fields"].items()}, "additionalProperties": False}
    Draft202012Validator.check_schema(converted)
    return schema, Draft202012Validator(converted, format_checker=FormatChecker())


def cu_value(field: JsonObject) -> object:
    """Unwrap only typed CU values, omitting absent object properties as in the notebook."""
    if not isinstance(field, dict):
        raise TypeError("CU fields must be dictionaries")
    kind = field.get("type")
    if not field:
        return None
    if kind == "array":
        items = field.get("valueArray")
        if items is None:
            return None
        if not isinstance(items, list):
            raise TypeError("valueArray must be a list of CU field objects")
        return [cu_value(item) for item in items]
    if kind == "object":
        properties = field.get("valueObject")
        if properties is None:
            return None
        if not isinstance(properties, dict):
            raise TypeError("valueObject must contain CU field objects")
        values = {name: cu_value(child) for name, child in properties.items()}
        return {name: value for name, value in values.items() if value is not None}
    if kind not in VALUE_KEYS:
        raise ValueError(f"Unsupported CU field type: {kind!r}")
    return field.get(VALUE_KEYS[kind])


def source_pages(field: JsonObject) -> set[int]:
    """Keep populated children without provenance unresolved unless their parent supplies it."""
    source = field.get("source") or ""
    if not isinstance(source, str):
        raise TypeError("CU source must be a string when present")
    pages = {int(number) for number in re.findall(r"D\(([1-9][0-9]*),", source)}
    children = field.get("valueArray", []) if field.get("type") == "array" else list((field.get("valueObject") or {}).values()) if field.get("type") == "object" else []
    for child in children or []:
        numbers = source_pages(child)
        if not source and cu_value(child) is not None and not numbers:
            return set()
        pages.update(numbers)
    return pages


def collect_targets(fields: JsonObject, definitions: JsonObject, *, prefix: str = "", parents: list[JsonObject] | None = None, pointer_prefix: str = "") -> list[JsonObject]:
    """Retain schema leaves, absent values, original array indices, and full ancestor rules."""
    targets: list[JsonObject] = []
    for name, definition in definitions.items():
        field = fields.get(name, {})
        if not isinstance(field, dict):
            raise TypeError(f"CU field {name!r} must be an object")
        path = f"{prefix}.{name}" if prefix else name
        field_pointer = pointer(pointer_prefix, name)
        ancestry = parents or []
        parent_rule = {key: definition[key] for key in ("type", "method", "description") if key in definition}
        if definition["type"] == "array" and field.get("type") == "array" and field.get("valueArray"):
            item_schema = definition["items"]
            if item_schema["type"] != "object":
                raise ValueError("Online page evaluation supports arrays of object records")
            for index, item in enumerate(field["valueArray"]):
                if item.get("type") != "object" or not isinstance(item.get("valueObject"), dict):
                    raise ValueError(f"Array item {path}[{index}] must have a CU object value")
                targets.extend(collect_targets(item["valueObject"], item_schema["properties"], prefix=f"{path}[{index}]", parents=[*ancestry, parent_rule, {key: item_schema[key] for key in ("type", "method", "description") if key in item_schema}], pointer_prefix=pointer(field_pointer, str(index))))
        elif definition["type"] == "object" and field.get("type") == "object":
            targets.extend(collect_targets(field.get("valueObject") or {}, definition["properties"], prefix=path, parents=[*ancestry, parent_rule], pointer_prefix=field_pointer))
        else:
            targets.append({"path": path, "pointer": field_pointer, "field_schema": definition, "parent_schema": ancestry, "cu_field": field, "value": cu_value(field)})
    return targets