"""Validate canonical extraction values against the declared model."""

from __future__ import annotations

import json

from ..config import EvaluationConfig
from ..types import JsonObject


class SchemaEvaluator:
    """Validate original canonical values using the declared extraction model."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Accept the caller's single extraction schema contract."""
        self.config = config

    def __call__(self, *, response: str, **kwargs: object) -> JsonObject:
        """Return conformance, violation count, and escaped violation paths."""
        value = json.loads(response)
        paths = self.config.extraction_schema.violation_paths(value) if self.config.extraction_schema is not None else []
        return {"valid": int(not paths), "violations": len(paths), "paths": paths}