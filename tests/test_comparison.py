"""Verify declarative field comparator registration and result contracts."""

import json
from datetime import datetime
from decimal import Decimal
from typing import Annotated, cast

import pytest
from pydantic import BeforeValidator, Field, ValidationError, field_validator

from extractly import (
    CUAdapter,
    ExtractionEvaluator,
    OfflineEvaluationConfig,
    OfflineEvaluator,
    OnlineEvaluationConfig,
    OnlineEvaluator,
    SchemaEvaluator,
    TableRule,
)
from extractly.comparison import Comparison, EvaluationModel, field_comparator
from extractly.config import EvaluationConfig
from extractly.eval_runner import LocalRunner
from extractly.types import JsonObject


class Quantity(EvaluationModel):
    """Provide a reusable typed comparator with an arbitrary method name."""

    quantity: Decimal

    @staticmethod
    @field_comparator("quantity")
    def close_enough(expected: Decimal, actual: Decimal) -> Comparison:
        """Keep strict equality separate from a small relaxed difference."""
        return Comparison(expected == actual, abs(expected - actual) <= Decimal("0.1"))


def test_given_decorated_field_when_model_defined_then_callable_bound() -> None:
    """Resolve the field's function independently from its method name."""
    result = Quantity._comparators["quantity"](Decimal("1"), Decimal("1.05"))

    assert result == Comparison(exact=False, tolerant=True)


def test_given_unknown_field_when_model_defined_then_registration_rejected() -> None:
    """Reject registration typos before any evaluation work."""
    with pytest.raises(ValueError, match="Unknown comparator field"):
        class Invalid(EvaluationModel):
            """Declare an invalid comparator binding."""

            quantity: Decimal

            @staticmethod
            @field_comparator("quantitty")
            def arbitrary_name(expected: Decimal, actual: Decimal) -> Comparison:
                """Demonstrate a misspelled registration."""
                return Comparison(expected == actual, expected == actual)


def test_given_inherited_comparator_when_overridden_then_subclass_policy_used() -> None:
    """Allow an explicit subclass policy without changing the parent's binding."""
    class StrictQuantity(Quantity):
        """Override the parent's quantity comparator."""

        @staticmethod
        @field_comparator("quantity")
        def strict_only(expected: Decimal, actual: Decimal) -> Comparison:
            """Require equality for both metric families."""
            return Comparison(expected == actual, expected == actual)

    result = StrictQuantity._comparators["quantity"](Decimal("1"), Decimal("1.05"))

    assert result == Comparison(exact=False, tolerant=False)
    assert Quantity._comparators["quantity"](Decimal("1"), Decimal("1.05")).tolerant


@pytest.mark.parametrize(("exact", "tolerant", "error"), [(1, True, TypeError), (True, 1, TypeError), (True, False, ValueError)])
def test_given_invalid_result_when_constructed_then_contract_rejected(exact: bool, tolerant: bool, error: type[Exception]) -> None:
    """Do not coerce comparator flags or accept inconsistent strict agreement."""
    with pytest.raises(error):
        Comparison(exact, tolerant)


def test_given_model_when_configured_then_selection_and_serializable_contract_derived() -> None:
    """Retain executable models in Python copies and describe them safely in JSON."""
    config = EvaluationConfig(extraction_schema=Quantity)

    restored = EvaluationConfig.model_validate(config.model_dump())

    assert restored.extraction_schema is Quantity
    assert restored.selects_path("/segments/0/fields/quantity")
    assert not restored.selects_path("/segments/0/fields/unrelated")
    assert "close_enough" in config.model_dump_json()


@pytest.mark.parametrize("settings", [{"include_fields": ["/quantity"]}, {"fields": {"/quantity": {}}}, {"evaluation_model": Quantity}, {"extraction_schema": {"type": "object"}}])
def test_given_model_and_legacy_policy_when_configured_then_ambiguity_rejected(settings: dict[str, object]) -> None:
    """Require a single declaration rather than competing selection contracts."""
    with pytest.raises(ValueError):
        EvaluationConfig.model_validate({"extraction_schema": Quantity, **settings})


class DatedQuantity(Quantity):
    """Compare a user-selected date format alongside quantity."""

    deliveryDate: str | None = Field(default=None, pattern=r"^[0-9]{2}/[0-9]{2}/[0-9]{4}$")

    @staticmethod
    @field_comparator("deliveryDate")
    def calendar_agreement(expected: str, actual: str) -> Comparison:
        """Parse calendar dates without hard-coded engine date formats."""
        gold = datetime.strptime(expected, "%d/%m/%Y").date()
        try:
            predicted = datetime.strptime(actual, "%d/%m/%Y").date()
        except ValueError:
            return Comparison(False, False, "Invalid calendar date")
        return Comparison(gold == predicted, gold == predicted, "Compared DD/MM/YYYY dates")


class Delivery(EvaluationModel):
    """Select nested quantities and dates while leaving row identities unscored."""

    items: list[DatedQuantity]


def test_given_nested_model_when_compared_then_selected_typed_fields_and_reasons_scored() -> None:
    """Use registered comparators and ignore unrelated metadata and fields."""
    config = EvaluationConfig(extraction_schema=Delivery)
    gold = {"segments": [{"fields": {"items": [{"quantity": 100, "deliveryDate": "08/10/2026", "extra": "gold"}]}}]}
    predicted = {"segments": [{"fields": {"items": [{"quantity": "100.05", "deliveryDate": "08/10/2026", "extra": "wrong"}]}}]}

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=json.dumps(gold))

    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["tolerant"]["f1"] == 1
    assert result["exact_match"] == 0
    assert result["comparisons"]["/segments/0/fields/items/0/deliveryDate"]["reason"] == "Compared DD/MM/YYYY dates"


def test_given_reordered_model_rows_when_keys_excluded_then_identity_still_aligns() -> None:
    """Retain original identity values without including them in field metrics."""
    config = EvaluationConfig(extraction_schema=Delivery, tables={"/segments/*/fields/items": TableRule(keys=["id"])})
    rows = [{"id": "A", "quantity": 1}, {"id": "B", "quantity": 2}]
    gold = {"segments": [{"fields": {"items": rows}}]}
    predicted = {"segments": [{"fields": {"items": rows[::-1]}}]}

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=json.dumps(gold))

    assert result["exact_match"] == result["f1"] == 1
    assert set(result["field_counts"]) == {"/segments/*/fields/items/*/quantity"}


def test_given_model_date_format_when_wrong_then_schema_and_comparison_fail() -> None:
    """Do not parse a format that violates the caller's declared string layout."""
    config = EvaluationConfig(extraction_schema=Delivery)
    gold = {"segments": [{"fields": {"items": [{"quantity": 1, "deliveryDate": "08/10/2026"}]}}]}
    predicted = {"segments": [{"fields": {"items": [{"quantity": 1, "deliveryDate": "2026-10-08"}]}}]}

    schema = SchemaEvaluator(config)(response=json.dumps(predicted))
    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=json.dumps(gold))

    assert schema["paths"] == ["/segments/0/fields/items/0/deliveryDate"]
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)


def test_given_invalid_model_gold_when_compared_then_validation_raises() -> None:
    """Reject invalid gold even when the prediction omits the same field."""
    config = EvaluationConfig(extraction_schema=Quantity)
    gold = {"segments": [{"fields": {"quantity": "invalid"}}]}

    with pytest.raises(ValidationError):
        ExtractionEvaluator(config)(response=json.dumps(gold), ground_truth=json.dumps(gold))


def test_given_model_required_omission_when_adapted_then_only_selected_slots_graded(raw: JsonObject) -> None:
    """Keep missing selected slots and full row context within the grading budget."""
    config = EvaluationConfig(extraction_schema=Delivery)
    del raw["contents"][0]["fields"]["items"]["valueArray"][0]["valueObject"]["quantity"]
    adapter = CUAdapter(config)

    rows = adapter.field_rows(adapter.prepare(raw, "source"), max_fields_per_document=2)

    assert len(rows) == 2
    missing = next(row for row in rows if row["field_path"].endswith("/0/quantity"))
    assert json.loads(missing["response"]) == {"presence": "absent"}
    assert json.loads(missing["row_context"])["eventId"] == "A"
    assert "extraction_schema" in json.loads(missing["contract"])


def test_given_model_when_offline_run_then_comparators_survive_and_tolerant_counts_pool(raw: JsonObject) -> None:
    """Exercise the real local runner, configuration copy, metrics, and report serialization."""
    gold = {"segments": [{"fields": {"items": [{"quantity": 12000}, {"quantity": 8000}]}}]}
    raw["contents"][0]["fields"]["items"]["valueArray"][0]["valueObject"]["quantity"]["valueNumber"] = 12000.05
    config = OfflineEvaluationConfig(extraction_schema=Delivery)

    report = OfflineEvaluator(config, runner=LocalRunner()).evaluate(raw, document_id="model-sample", ground_truth=gold)

    assert report["metrics"]["extraction.micro_f1"] == 0.5
    assert report["metrics"]["extraction.tolerant_micro_f1"] == 1
    assert report["fields"]["/segments/*/fields/items/*/quantity"]["tolerant"]["tp"] == 2
    assert json.loads(json.dumps(report))["configuration_hash"] == report["configuration_hash"]


def test_given_model_when_online_run_then_schema_contract_remains_reference_free(raw: JsonObject) -> None:
    """Use model validation online without running gold-based comparison functions."""
    config = OnlineEvaluationConfig(extraction_schema=Delivery)

    report = OnlineEvaluator(config, runner=LocalRunner()).evaluate(raw, document_id="model-online")

    assert report["metrics"]["schema.valid"] == 1
    assert report["mode"] == "online"


def test_given_comparator_failure_when_compared_then_original_exception_propagates() -> None:
    """Never turn caller failures into successful or ordinary mismatch results."""
    failure = ValueError("Comparison failed")

    class Broken(EvaluationModel):
        """Demonstrate a caller-owned comparison failure."""

        value: int

        @staticmethod
        @field_comparator("value")
        def fails(expected: int, actual: int) -> Comparison:
            """Raise the original exception for the test."""
            raise failure

    config = EvaluationConfig(extraction_schema=Broken)
    gold = json.dumps({"segments": [{"fields": {"value": 1}}]})

    with pytest.raises(ValueError) as caught:
        ExtractionEvaluator(config)(response=gold, ground_truth=gold)

    assert caught.value is failure


def test_given_model_default_when_prediction_missing_then_value_not_invented() -> None:
    """Use defaults for validation optionality, never as observed extraction values."""
    class DefaultQuantity(Quantity):
        """Make quantity optional through a default value."""

        quantity: Decimal = Decimal("9")

    config = EvaluationConfig(extraction_schema=DefaultQuantity)
    gold = json.dumps({"segments": [{"fields": {"quantity": 9}}]})
    predicted = json.dumps({"segments": [{"fields": {}}]})

    result = ExtractionEvaluator(config)(response=predicted, ground_truth=gold)

    assert SchemaEvaluator(config)(response=predicted)["valid"] == 1
    assert (result["tp"], result["fp"], result["fn"]) == (0, 0, 1)


def test_given_alias_when_model_compiled_then_input_names_and_pointer_escapes_used() -> None:
    """Resolve registered Python field names to escaped CU property names."""
    class AliasedQuantity(Quantity):
        """Use a source property that requires JSON Pointer escaping."""

        quantity: Decimal = Field(alias="Net/Weight~")

    config = EvaluationConfig(extraction_schema=AliasedQuantity)
    gold = json.dumps({"segments": [{"fields": {"Net/Weight~": 1}}]})

    result = ExtractionEvaluator(config)(response=gold, ground_truth=gold)

    assert result["field_counts"] == {"/segments/*/fields/Net~1Weight~0": [1, 0, 0]}


@pytest.mark.parametrize("prediction", ["invalid", {"nested": 1}, True])
def test_given_invalid_model_prediction_when_scored_then_directional_errors_preserved(prediction: object) -> None:
    """Count invalid present predictions as errors rather than dropping their fields."""
    config = EvaluationConfig(extraction_schema=Quantity)
    gold = json.dumps({"segments": [{"fields": {"quantity": 1}}]})
    predicted = json.dumps({"segments": [{"fields": {"quantity": prediction}}]})

    result = ExtractionEvaluator(config)(response=predicted, ground_truth=gold)

    assert SchemaEvaluator(config)(response=predicted)["valid"] == 0
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)


@pytest.mark.parametrize(("quantities", "fp", "fn"), [([1], 0, 1), ([1, 2, 2], 1, 0)])
def test_given_model_row_count_change_when_scored_then_multiplicity_preserved(quantities: list[int], fp: int, fn: int) -> None:
    """Retain directional counts for missing and extra selected row occurrences."""
    config = EvaluationConfig(extraction_schema=Delivery)
    gold = json.dumps({"segments": [{"fields": {"items": [{"quantity": 1}, {"quantity": 2}]}}]})
    predicted = json.dumps({"segments": [{"fields": {"items": [{"quantity": value} for value in quantities]}}]})

    result = ExtractionEvaluator(config)(response=predicted, ground_truth=gold)

    assert (result["fp"], result["fn"]) == (fp, fn)
    assert result["exact_match"] == 0


def test_given_annotated_transform_when_configured_then_unsupported_policy_rejected() -> None:
    """Reject transformations hidden inside Annotated rather than scoring changed data."""
    class Transforming(EvaluationModel):
        """Declare an unsupported Pydantic transformation."""

        value: Annotated[str, BeforeValidator(str.strip)]

    with pytest.raises(ValueError, match="Annotated validators"):
        EvaluationConfig(extraction_schema=Transforming)


def test_given_field_validator_when_defined_then_unsupported_policy_rejected() -> None:
    """Reject custom Pydantic validators before any compilation or evaluation."""
    with pytest.raises(ValueError, match="custom Pydantic validators"):
        class Transforming(EvaluationModel):
            """Declare a custom validator that would change extracted values."""

            value: str

            @field_validator("value")
            @classmethod
            def strip_value(cls, value: str) -> str:
                """Demonstrate a transformation that belongs in a comparator."""
                return value.strip()


def test_given_bad_comparator_return_when_invoked_then_contract_error_raised() -> None:
    """Validate runtime output even if caller annotations claim the correct type."""
    class BadResult(EvaluationModel):
        """Declare a comparator with a malformed return value."""

        value: int

        @staticmethod
        @field_comparator("value")
        def invalid_return(expected: int, actual: int) -> Comparison:
            """Return an invalid object to exercise the boundary check."""
            return cast(Comparison, True)

    config = EvaluationConfig(extraction_schema=BadResult)
    gold = json.dumps({"segments": [{"fields": {"value": 1}}]})

    with pytest.raises(TypeError, match="must return Comparison"):
        ExtractionEvaluator(config)(response=gold, ground_truth=gold)