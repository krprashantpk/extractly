"""Exercise stateless schema validation with metric-shaped results."""

from copy import deepcopy

import pytest
from pydantic import BaseModel, Field

from extractly.evaluators.schema import SchemaEvaluator
from extractly.types import JsonObject


class Extraction(BaseModel):
    """Declare a constrained aliased field."""

    quantity: int = Field(alias="net/weight~", ge=0)


@pytest.mark.parametrize(("candidate", "valid", "paths"), [({"net/weight~": 0}, 1, []), ({"net/weight~": -1}, 0, ["/net~1weight~0"]), ({}, 0, ["/net~1weight~0"])])
def test_given_candidate_when_validated_then_metrics_and_original_paths_are_returned(candidate: JsonObject, valid: int, paths: list[str]) -> None:
    """Report validation failures without changing caller-owned evidence."""
    original = deepcopy(candidate)

    result = SchemaEvaluator()(candidate, Extraction)

    assert result == {"metrics": {"valid": valid, "violations": len(paths)}, "paths": paths}
    assert candidate == original