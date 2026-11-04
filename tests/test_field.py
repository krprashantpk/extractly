"""Exercise standalone deterministic field scoring without configuration or workflows."""

from copy import deepcopy
from decimal import Decimal
from typing import cast

import pytest
from pydantic import BaseModel, Field, ValidationError

from extractly.comparison import EvaluationModel, field_comparator
from extractly.evaluators.field import FieldEvaluator


class Quantity(EvaluationModel):
    """Compare an aliased quantity using a caller-owned tolerance policy."""

    quantity: Decimal = Field(alias="net/weight~", ge=0)
    note: str | None = None

    @staticmethod
    @field_comparator("quantity")
    def close_enough(expected: Decimal, actual: Decimal) -> bool:
        """Accept quantity differences of at most one tenth."""
        return abs(expected - actual) <= Decimal("0.1")


class Delivery(EvaluationModel):
    """Select quantity rows without scoring contextual identifiers."""

    items: list[Quantity]


def test_given_aliased_values_when_called_then_validated_comparators_and_scope_are_preserved() -> None:
    """Score selected fields and retain inputs without configuration or canonical wrappers."""
    candidate = {"items": [{"net/weight~": "1.05", "ignored": "wrong"}]}
    original = deepcopy(candidate)
    gold = {"items": [{"net/weight~": 1, "ignored": "reference"}]}

    result = FieldEvaluator()(candidate, gold, Delivery)

    assert result["metrics"] == {"extraction.document.tp": 1, "extraction.document.fp": 0, "extraction.document.fn": 0, "extraction.document.precision": 1.0, "extraction.document.recall": 1.0, "extraction.document.f1": 1.0}
    assert result["field_counts"] == {"/items/*/net~1weight~0": [1, 0, 0], "/items/*/note": [0, 0, 0]}
    assert result["comparisons"]["/items/0/net~1weight~0"]["match"] is True
    assert result["exact_match"] == 1
    assert candidate == original


@pytest.mark.parametrize("actual", [2, "invalid", -1, {"wrong": "container"}])
def test_given_wrong_present_value_when_called_then_counts_both_fp_and_fn(actual: object) -> None:
    """Keep invalid or incorrect predictions as directional errors rather than raising."""
    result = FieldEvaluator()({"net/weight~": actual}, {"net/weight~": 1}, Quantity)

    assert [result["metrics"][f"extraction.document.{name}"] for name in ("tp", "fp", "fn")] == [0, 1, 1]
    assert result["exact_match"] == 0


def test_given_missing_defaulted_field_when_called_then_no_observed_value_is_invented() -> None:
    """Treat a candidate omission as missing even when the model declares a default."""
    class Defaulted(EvaluationModel):
        """Declare a default that must not become extracted evidence."""

        value: int = 9

    result = FieldEvaluator()({}, {"value": 9}, Defaulted)

    assert result["field_counts"] == {"/value": [0, 0, 1]}
    assert result["comparisons"]["/value"]["status"] == "missing"


def test_given_absence_zero_and_false_when_called_then_presence_is_not_truthiness() -> None:
    """Distinguish real zero and false values from missing, null, and empty strings."""
    class Values(EvaluationModel):
        """Declare optional text alongside falsy present values."""

        zero: int
        flag: bool
        note: str | None = None

    result = FieldEvaluator()({"zero": 0, "flag": False, "note": ""}, {"zero": 0, "flag": False}, Values)

    assert result["metrics"]["extraction.document.tp"] == 2
    assert result["field_counts"]["/note"] == [0, 0, 0]
    assert result["exact_match"] == 1


@pytest.mark.parametrize(("values", "expected"), [([2, 1], [0, 2, 2]), ([1], [1, 0, 1]), ([1, 2, 3], [2, 1, 0])])
def test_given_changed_rows_when_called_then_fields_remain_positional(values: list[int], expected: list[int]) -> None:
    """Preserve positional field errors without emitting structural metrics."""
    candidate = {"items": [{"net/weight~": value} for value in values]}
    gold = {"items": [{"net/weight~": 1}, {"net/weight~": 2}]}

    result = FieldEvaluator()(candidate, gold, Delivery)

    assert result["field_counts"]["/items/*/net~1weight~0"] == expected
    assert not any("table" in name for name in result["metrics"])


def test_given_invalid_gold_when_called_then_validation_raises() -> None:
    """Reject invalid reference values before counting predictions."""
    with pytest.raises(ValidationError):
        FieldEvaluator()({"net/weight~": 1}, {"net/weight~": "invalid"}, Quantity)


def test_given_comparator_failure_when_called_then_original_exception_propagates() -> None:
    """Keep caller comparison failures visible instead of reporting a mismatch."""
    failure = RuntimeError("Caller comparison failed")

    class Broken(EvaluationModel):
        """Declare a comparison function that raises its own exception."""

        value: int

        @staticmethod
        @field_comparator("value")
        def compare(expected: int, actual: int) -> bool:
            """Raise the original caller-owned failure."""
            raise failure

    with pytest.raises(RuntimeError) as caught:
        FieldEvaluator()({"value": 1}, {"value": 1}, Broken)

    assert caught.value is failure


def test_given_plain_model_when_called_then_comparison_model_is_required() -> None:
    """Require the model type that owns registered comparison policies."""
    with pytest.raises(TypeError, match="EvaluationModel"):
        FieldEvaluator()({}, {}, cast(type[EvaluationModel], BaseModel))