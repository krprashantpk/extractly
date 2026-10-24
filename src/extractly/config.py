"""Validate comparison policy without loading application configuration."""

from __future__ import annotations

import os
import re
from decimal import Decimal, InvalidOperation
from typing import Literal
from urllib.parse import urlsplit

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from jsonschema.validators import validator_for
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .types import JsonObject


class Settings(BaseModel):
    """Reject unknown settings, type coercion, and nonfinite numeric policy."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, validate_assignment=True)


def _validate_pointer(path: str) -> None:
    """Reject malformed nonempty JSON Pointer patterns."""
    if not path.startswith("/") or any(part == "" for part in path[1:].split("/")) or re.search(r"~(?![01])", path):
        raise ValueError("Field/table paths must be nonempty JSON Pointer patterns with valid escapes")


class FieldRule(Settings):
    """Define typed comparison for a JSON Pointer pattern.

    An asterisk addresses existing array occurrences, not undiscovered rows.
    Numeric strings and unit conversions require explicit opt-in policy.
    """

    kind: Literal["exact", "decimal", "date", "text"] = "exact"
    required: bool = False
    description: str = ""
    absolute_tolerance: str = "0"
    relative_tolerance: str = "0"
    date_format: str = "%Y-%m-%d"
    unit_field: str | None = None
    unit_scales: dict[str, str] = Field(default_factory=dict)
    coerce_numeric_strings: bool = False

    @model_validator(mode="after")
    def finite_tolerances(self) -> FieldRule:
        """Reject invalid decimal and unit policy before provider calls."""
        try:
            for value in (self.absolute_tolerance, self.relative_tolerance, *self.unit_scales.values()):
                if not Decimal(value).is_finite() or Decimal(value) < 0:
                    raise ValueError("Tolerances and conversion factors must be finite and nonnegative")
            if any(Decimal(value) == 0 for value in self.unit_scales.values()):
                raise ValueError("Unit conversion factors must be positive")
        except InvalidOperation as exc:
            raise ValueError("Invalid decimal comparison policy") from exc
        if self.unit_scales and not self.unit_field:
            raise ValueError("Unit conversion requires an explicit unit_field")
        if self.unit_field and not self.unit_scales:
            raise ValueError("unit_field requires explicit unit_scales")
        if self.unit_field and self.unit_field.startswith("/"):
            _validate_pointer(self.unit_field)
        return self


class TableRule(Settings):
    """Align occurrences by independent keys or explicitly positional ordering."""

    keys: list[str] = Field(default_factory=list)
    level: Literal["row", "instance"] = "row"

    @model_validator(mode="after")
    def valid_keys(self) -> TableRule:
        """Reject empty or wildcard row identities."""
        if len(set(self.keys)) != len(self.keys):
            raise ValueError("Table identity keys must be unique")
        for key in self.keys:
            if not key or "*" in key:
                raise ValueError("Table identity keys must be nonempty and cannot contain wildcards")
            if key.startswith("/"):
                _validate_pointer(key)
        return self


class ModelSettings(Settings):
    """Describe a model without owning credentials, retries, or client cleanup."""

    azure_endpoint: str = Field(min_length=1)
    azure_deployment: str = Field(min_length=1)
    api_version: str = "2025-04-01-preview"
    api_key_env: str | None = None
    is_reasoning_model: bool = False

    @model_validator(mode="after")
    def valid_endpoint(self) -> ModelSettings:
        """Disallow embedded credentials and endpoint query parameters."""
        url = urlsplit(self.azure_endpoint)
        if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("The model endpoint must be HTTPS without credentials, query or fragment")
        if not self.azure_deployment.strip():
            raise ValueError("azure_deployment must not be blank")
        if self.api_key_env is not None and not self.api_key_env.strip():
            raise ValueError("api_key_env must name a nonempty environment variable")
        return self

    def configuration(self) -> AzureOpenAIModelConfiguration:
        """Resolve the explicitly named key only at the SDK call boundary.

        Raises:
            ValueError: The configured API-key environment variable is absent or empty.
        """
        result = AzureOpenAIModelConfiguration(azure_endpoint=self.azure_endpoint, azure_deployment=self.azure_deployment, api_version=self.api_version)
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise ValueError("Configured model API-key environment variable is absent or empty")
            result["api_key"] = key
        return result


class EvaluationConfig(Settings):
    """Share the extraction contract, context budget, and minimum-score gates.

    Workflow-specific settings belong to OfflineEvaluationConfig or
    OnlineEvaluationConfig, not this reusable adapter/evaluator contract.
    """

    name: str = Field(default="extractly", min_length=1)
    fields: dict[str, FieldRule] = Field(default_factory=dict)
    tables: dict[str, TableRule] = Field(default_factory=dict)
    extraction_schema: JsonObject = Field(default_factory=dict)
    project: str | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    max_context_characters: int = Field(default=100000, ge=1)

    @model_validator(mode="after")
    def consistent_policy(self) -> EvaluationConfig:
        """Validate local schemas, pointers, and shared extraction policy."""
        validator_for(self.extraction_schema).check_schema(self.extraction_schema)

        def local_references(value: object) -> None:
            """Disallow network resolution, including dynamic and recursive references."""
            if isinstance(value, dict):
                for key in ("$ref", "$dynamicRef", "$recursiveRef"):
                    if key in value and (not isinstance(value[key], str) or not value[key].startswith("#")):
                        raise ValueError("Schemas must use local references; remote fetching is not enabled")
                for child in value.values():
                    local_references(child)
            elif isinstance(value, list):
                for child in value:
                    local_references(child)

        local_references(self.extraction_schema)
        for path in [*self.fields, *self.tables]:
            _validate_pointer(path)
        for path, rule in self.fields.items():
            if rule.unit_field and rule.unit_field.count("*") > path.count("*"):
                raise ValueError("A unit reference cannot introduce additional array wildcards")
        if any(not name.strip() for name in self.thresholds):
            raise ValueError("Threshold metric names must not be blank")
        return self
