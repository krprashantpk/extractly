"""Preserve caller-defined static comparison policies independently of pipelines."""

from decimal import Decimal
from typing import Annotated, cast

import pytest
from pydantic import BeforeValidator, Field, field_validator

from extractly import EvaluationModel, FieldEvaluator, field_comparator
from extractly.comparison import ModelContract


class Quantity(EvaluationModel):
    """Declare a reusable decimal tolerance policy."""

    quantity: Decimal

    @staticmethod
    @field_comparator("quantity")
    def close_enough(expected: Decimal, actual: Decimal) -> bool:
        """Accept quantities within one tenth."""
        return abs(expected - actual) <= Decimal("0.1")


def test_given_inherited_comparator_when_overridden_then_parent_policy_is_unchanged() -> None:
    """Keep static comparison registration reusable across model subclasses."""
    class StrictQuantity(Quantity):
        """Replace the inherited tolerance with exact equality."""

        @staticmethod
        @field_comparator("quantity")
        def exact(expected: Decimal, actual: Decimal) -> bool:
            """Require equal quantities."""
            return expected == actual

    tolerant = FieldEvaluator()({"quantity": "1.05"}, {"quantity": 1}, Quantity)
    strict = FieldEvaluator()({"quantity": "1.05"}, {"quantity": 1}, StrictQuantity)

    assert tolerant["exact_match"] == 1
    assert strict["exact_match"] == 0


def test_given_unknown_field_when_comparator_registered_then_declaration_is_rejected() -> None:
    """Reject comparison policy typos during model definition."""
    with pytest.raises(ValueError, match="Unknown comparator field"):
        class Invalid(EvaluationModel):
            """Demonstrate an invalid field binding."""

            value: int

            @staticmethod
            @field_comparator("typo")
            def compare(expected: int, actual: int) -> bool:
                """Compare values for an invalid declaration."""
                return expected == actual


@pytest.mark.parametrize("invalid", [None, 0, 1, "true", {}, (True, True)])
def test_given_nonboolean_comparator_result_when_called_then_contract_error_is_visible(invalid: object) -> None:
    """Do not coerce caller comparison results into successful matches."""
    class InvalidResult(EvaluationModel):
        """Return a malformed result despite the annotated signature."""

        value: int

        @staticmethod
        @field_comparator("value")
        def compare(expected: int, actual: int) -> bool:
            """Expose the malformed caller-owned comparison result."""
            return cast(bool, invalid)

    with pytest.raises(TypeError, match="return bool"):
        FieldEvaluator()({"value": 1}, {"value": 1}, InvalidResult)


def test_given_aliased_field_when_compiled_then_original_canonical_paths_remain_available() -> None:
    """Preserve comparison contracts used by client-side CU adapter helpers."""
    class Aliased(Quantity):
        """Use a field name requiring pointer escaping."""

        quantity: Decimal = Field(alias="net/weight~")

    contract = ModelContract.compile(Aliased)

    assert set(contract.fields) == {"/segments/*/fields/net~1weight~0"}
    assert "close_enough" in str(contract.describe())


def test_given_transforming_validator_when_compiled_then_policy_is_rejected() -> None:
    """Require explicit comparators instead of silently transformed evidence."""
    class Transforming(EvaluationModel):
        """Declare an unsupported annotation-level transformation."""

        value: Annotated[str, BeforeValidator(str.strip)]

    with pytest.raises(ValueError, match="Annotated validators"):
        ModelContract.compile(Transforming)


def test_given_custom_pydantic_validator_when_declared_then_transformation_is_rejected() -> None:
    """Keep field comparison methods distinct from value-transformation validators."""
    with pytest.raises(ValueError, match="custom Pydantic validators"):
        class Transforming(EvaluationModel):
            """Declare an unsupported custom transformation."""

            value: str

            @field_validator("value")
            @classmethod
            def trim(cls, value: str) -> str:
                """Demonstrate a transformation that changes observed values."""
                return value.strip()