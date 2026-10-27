"""Validate comparison policy without loading application configuration."""

from __future__ import annotations

import os
import re
from typing import Literal
from urllib.parse import urlsplit

from azure.ai.evaluation import AzureOpenAIModelConfiguration
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_serializer, model_validator

from .comparison import EvaluationModel, ModelContract
from .types import JsonObject


class Settings(BaseModel):
    """Reject unknown settings, type coercion, and nonfinite numeric policy."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False, validate_assignment=True)


def _validate_pointer(path: str) -> None:
    """Reject malformed nonempty JSON Pointer patterns."""
    if not path.startswith("/") or any(part == "" for part in path[1:].split("/")) or re.search(r"~(?![01])", path):
        raise ValueError("Field/table paths must be nonempty JSON Pointer patterns with valid escapes")


class TableRule(Settings):
    """Select table-row or table-count metrics for positionally compared arrays."""

    level: Literal["table_row", "table_count"] = "table_row"


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

    extraction_model is the single model declaration for field selection,
    Pydantic validation, and registered comparison functions within each segment's
    fields object. None evaluates all emitted fields using default equality.
    Tables select structural metrics; all arrays are compared in positional order.
    Workflow-specific settings belong to OfflineEvaluationConfig or
    OnlineEvaluationConfig, not this reusable adapter/evaluator contract.
    """

    name: str = Field(default="extractly", min_length=1)
    extraction_model: type[EvaluationModel] | None = None
    tables: dict[str, TableRule] = Field(default_factory=dict)
    project: str | None = None
    thresholds: dict[str, float] = Field(default_factory=dict)
    max_context_characters: int = Field(default=100000, ge=1)
    _extraction_schema: ModelContract | None = PrivateAttr(default=None)

    @property
    def extraction_schema(self) -> ModelContract | None:
        """Return the compiled model policy without altering caller-owned values."""
        return self._extraction_schema

    @field_serializer("extraction_model", when_used="json")
    def serialize_extraction_model(self, model: type[EvaluationModel] | None) -> JsonObject | None:
        """Describe executable policy in JSON while retaining classes in Python dumps."""
        return self._extraction_schema.describe() if model is not None and self._extraction_schema is not None else None

    @model_validator(mode="after")
    def consistent_policy(self) -> EvaluationConfig:
        """Compile the extraction model and validate table paths and score gates."""
        self._extraction_schema = ModelContract.compile(self.extraction_model) if self.extraction_model is not None else None
        for path in self.tables:
            _validate_pointer(path)
        if any(not name.strip() for name in self.thresholds):
            raise ValueError("Threshold metric names must not be blank")
        return self

    def selects_path(self, path: str, *, include_ancestors: bool = False) -> bool:
        """Check selection without altering original source values.

        Args:
            path: Escaped JSON Pointer for an occurrence in the original document.
            include_ancestors: Include containers that must be traversed to reach selected fields.

        Returns:
            Whether a selector covers this path or, when requested, a descendant; a selected container includes its descendants and an asterisk matches one path component.
        """
        if self._extraction_schema is None:
            return True
        parts = path.split("/")[1:] if path else []
        return any(
            (include_ancestors or len(selected) <= len(parts)) and all(expected == "*" or expected == actual for expected, actual in zip(selected, parts, strict=False))
            for selector in self._extraction_schema.fields
            for selected in [selector.split("/")[1:]]
        )
