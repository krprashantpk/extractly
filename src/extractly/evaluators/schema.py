"""Validate normalized extraction values against a caller-supplied model."""

from __future__ import annotations

from pydantic import BaseModel, ValidationError

from ..types import JsonObject


class SchemaEvaluator:
    """Validate observed values without retaining configuration or filling inputs.

    Metrics:
        valid: Integer 1 when the candidate conforms to the model, otherwise 0.
        violations: Number of Pydantic validation errors in the candidate.
    """

    def __call__(self, extracted_values: JsonObject, extraction_model: type[BaseModel]) -> JsonObject:
        """Return conformance metrics and escaped validation-error paths.

        Args:
            extracted_values: Normalized candidate dictionary, never modified.
            extraction_model: Pydantic model class defining validation rules.

        Returns:
            A metrics dictionary containing valid and violations, plus diagnostic paths.

        Raises:
            TypeError: The candidate is not a dictionary or the model is not a Pydantic model class.
        """
        if not isinstance(extracted_values, dict):
            raise TypeError("extracted_values must be a dictionary")
        if not isinstance(extraction_model, type) or not issubclass(extraction_model, BaseModel):
            raise TypeError("extraction_model must be a Pydantic BaseModel class")
        paths: list[str] = []
        try:
            extraction_model.model_validate(extracted_values)
        except ValidationError as error:
            paths = ["".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in item["loc"]) for item in error.errors()]
        return {"metrics": {"valid": int(not paths), "violations": len(paths)}, "paths": paths}