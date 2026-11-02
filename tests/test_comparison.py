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
from extractly.comparison import EvaluationModel, field_comparator
from extractly.config import EvaluationConfig
from extractly.eval_runner import LocalRunner
from extractly.types import JsonObject


class Quantity(EvaluationModel):
    """Provide a reusable typed comparator with an arbitrary method name."""

    quantity: Decimal

    @staticmethod
    @field_comparator("quantity")
    def close_enough(expected: Decimal, actual: Decimal) -> bool:
        """Accept values whose difference is at most 0.1."""
        return abs(expected - actual) <= Decimal("0.1")


@pytest.mark.parametrize(("predicted_quantity", "matched"), [(100, True), (101, True), (99, True), (102, False), (98, False)])
def test_given_unit_tolerance_when_compared_then_boolean_match_drives_scores(predicted_quantity: int, matched: bool) -> None:
    """Count allowed variation as an ordinary match, not a separate relaxed score."""
    class UnitQuantity(EvaluationModel):
        """Accept differences of at most one unit through a boolean comparator."""

        quantity: int

        @staticmethod
        @field_comparator("quantity")
        def within_one_unit(expected: int, actual: int) -> bool:
            """Decide whether the values match under the allowed variation."""
            return abs(expected - actual) <= 1

    gold = {"segments": [{"fields": {"quantity": 100}}]}
    predicted = {"segments": [{"fields": {"quantity": predicted_quantity}}]}

    result = ExtractionEvaluator(EvaluationConfig(extraction_model=UnitQuantity))(response=json.dumps(predicted), ground_truth=json.dumps(gold))

    assert (result["tp"], result["fp"], result["fn"]) == ((1, 0, 0) if matched else (0, 1, 1))
    assert result["precision"] == result["recall"] == result["f1"] == float(matched)
    assert result["exact_match"] == int(matched)
    assert result["comparisons"]["/segments/0/fields/quantity"]["match"] is matched
    assert not any("tolerant" in key for key in result)


def test_given_unknown_field_when_model_defined_then_registration_rejected() -> None:
    """Reject registration typos before any evaluation work."""
    with pytest.raises(ValueError, match="Unknown comparator field"):
        class Invalid(EvaluationModel):
            """Declare an invalid comparator binding."""

            quantity: Decimal

            @staticmethod
            @field_comparator("quantitty")
            def arbitrary_name(expected: Decimal, actual: Decimal) -> bool:
                """Demonstrate a misspelled registration."""
                return expected == actual


def test_given_inherited_comparator_when_overridden_then_subclass_policy_used() -> None:
    """Allow an explicit subclass policy without changing the parent's binding."""
    class StrictQuantity(Quantity):
        """Override the parent's quantity comparator."""

        @staticmethod
        @field_comparator("quantity")
        def strict_only(expected: Decimal, actual: Decimal) -> bool:
            """Require equality instead of the parent's allowed variation."""
            return expected == actual

    result = StrictQuantity._comparators["quantity"](Decimal("1"), Decimal("1.05"))

    assert result is False
    assert Quantity._comparators["quantity"](Decimal("1"), Decimal("1.05")) is True


def test_given_model_when_configured_then_selection_and_serializable_contract_derived() -> None:
    """Retain executable models in Python copies and describe them safely in JSON."""
    config = EvaluationConfig(extraction_model=Quantity)

    restored = EvaluationConfig.model_validate(config.model_dump())

    assert restored.extraction_model is Quantity
    assert restored.selects_path("/segments/0/fields/quantity")
    assert not restored.selects_path("/segments/0/fields/unrelated")
    assert "close_enough" in config.model_dump_json()


@pytest.mark.parametrize("settings", [{"include_fields": ["/quantity"]}, {"fields": {"/quantity": {}}}, {"evaluation_model": Quantity}, {"extraction_model": {"type": "object"}}, {"extraction_schema": Quantity}])
def test_given_model_and_legacy_policy_when_configured_then_ambiguity_rejected(settings: dict[str, object]) -> None:
    """Require a single declaration rather than competing selection contracts."""
    with pytest.raises(ValueError):
        EvaluationConfig.model_validate({"extraction_model": Quantity, **settings})


class DatedQuantity(Quantity):
    """Compare a user-selected date format alongside quantity."""

    deliveryDate: str | None = Field(default=None, pattern=r"^[0-9]{2}/[0-9]{2}/[0-9]{4}$")

    @staticmethod
    @field_comparator("deliveryDate")
    def calendar_agreement(expected: str, actual: str) -> bool:
        """Parse calendar dates without hard-coded engine date formats."""
        gold = datetime.strptime(expected, "%d/%m/%Y").date()
        try:
            predicted = datetime.strptime(actual, "%d/%m/%Y").date()
        except ValueError:
            return False
        return gold == predicted


class Delivery(EvaluationModel):
    """Select nested quantities and dates while leaving row identities unscored."""

    items: list[DatedQuantity]


def test_given_nested_model_when_compared_then_selected_typed_fields_and_matches_scored() -> None:
    """Use registered comparators and ignore unrelated metadata and fields."""
    config = EvaluationConfig(extraction_model=Delivery)
    gold = {"segments": [{"fields": {"items": [{"quantity": 100, "deliveryDate": "08/10/2026", "extra": "gold"}]}}]}
    predicted = {"segments": [{"fields": {"items": [{"quantity": "100.05", "deliveryDate": "08/10/2026", "extra": "wrong"}]}}]}

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=json.dumps(gold))

    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
    assert result["f1"] == result["exact_match"] == 1
    assert result["comparisons"]["/segments/0/fields/items/0/deliveryDate"]["match"] is True


def test_given_reordered_model_rows_when_identities_excluded_then_positional_errors() -> None:
    """Compare selected values at their original indices regardless of identities."""
    config = EvaluationConfig(extraction_model=Delivery, tables={"/segments/*/fields/items": TableRule()})
    rows = [{"id": "A", "quantity": 1}, {"id": "B", "quantity": 2}]
    gold = {"segments": [{"fields": {"items": rows}}]}
    predicted = {"segments": [{"fields": {"items": rows[::-1]}}]}

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=json.dumps(gold))

    assert result["exact_match"] == result["f1"] == 0
    assert result["table_row_tp"] == 2
    assert set(result["field_counts"]) == {"/segments/*/fields/items/*/quantity"}


def test_given_model_date_format_when_wrong_then_schema_and_comparison_fail() -> None:
    """Do not parse a format that violates the caller's declared string layout."""
    config = EvaluationConfig(extraction_model=Delivery)
    gold = {"segments": [{"fields": {"items": [{"quantity": 1, "deliveryDate": "08/10/2026"}]}}]}
    predicted = {"segments": [{"fields": {"items": [{"quantity": 1, "deliveryDate": "2026-10-08"}]}}]}

    schema = SchemaEvaluator(config)(response=json.dumps(predicted))
    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=json.dumps(gold))

    assert schema["paths"] == ["/segments/0/fields/items/0/deliveryDate"]
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)


def test_given_invalid_model_gold_when_compared_then_validation_raises() -> None:
    """Reject invalid gold even when the prediction omits the same field."""
    config = EvaluationConfig(extraction_model=Quantity)
    gold = {"segments": [{"fields": {"quantity": "invalid"}}]}

    with pytest.raises(ValidationError):
        ExtractionEvaluator(config)(response=json.dumps(gold), ground_truth=json.dumps(gold))


def test_given_model_required_omission_when_adapted_then_only_selected_slots_graded(raw: JsonObject) -> None:
    """Keep missing selected slots and full row context within the grading budget."""
    config = EvaluationConfig(extraction_model=Delivery)
    del raw["contents"][0]["fields"]["items"]["valueArray"][0]["valueObject"]["quantity"]
    adapter = CUAdapter(config)

    rows = adapter.field_rows(adapter.prepare(raw, "source"), max_fields_per_document=2)

    assert len(rows) == 2
    missing = next(row for row in rows if row["field_path"].endswith("/0/quantity"))
    assert json.loads(missing["response"]) == {"presence": "absent"}
    assert json.loads(missing["row_context"])["eventId"] == "A"
    assert "extraction_model" in json.loads(missing["contract"])


def test_given_model_when_offline_run_then_boolean_comparator_matches_pool(raw: JsonObject) -> None:
    """Exercise the real local runner, configuration copy, metrics, and report serialization."""
    gold = {"segments": [{"fields": {"items": [{"quantity": 12000}, {"quantity": 8000}]}}]}
    raw["contents"][0]["fields"]["items"]["valueArray"][0]["valueObject"]["quantity"]["valueNumber"] = 12000.05
    config = OfflineEvaluationConfig(extraction_model=Delivery)

    report = OfflineEvaluator(config, runner=LocalRunner()).evaluate(raw, document_id="model-sample", ground_truth=gold)

    assert report["metrics"]["extraction.pooled_field_f1"] == 1
    assert report["metrics"]["extraction.exact_match"] == 1
    assert report["metrics"]["extraction.document_exact_match_rate"] == 1
    assert report["fields"]["/segments/*/fields/items/*/quantity"]["tp"] == 2
    assert not any("tolerant" in key for key in report["metrics"])
    assert json.loads(json.dumps(report))["configuration_hash"] == report["configuration_hash"]


def test_given_model_when_online_run_then_schema_contract_remains_reference_free(raw: JsonObject) -> None:
    """Use model validation online without running gold-based comparison functions."""
    config = OnlineEvaluationConfig(extraction_model=Delivery)

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
        def fails(expected: int, actual: int) -> bool:
            """Raise the original exception for the test."""
            raise failure

    config = EvaluationConfig(extraction_model=Broken)
    gold = json.dumps({"segments": [{"fields": {"value": 1}}]})

    with pytest.raises(ValueError) as caught:
        ExtractionEvaluator(config)(response=gold, ground_truth=gold)

    assert caught.value is failure


def test_given_model_default_when_prediction_missing_then_value_not_invented() -> None:
    """Use defaults for validation optionality, never as observed extraction values."""
    class DefaultQuantity(Quantity):
        """Make quantity optional through a default value."""

        quantity: Decimal = Decimal("9")

    config = EvaluationConfig(extraction_model=DefaultQuantity)
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

    config = EvaluationConfig(extraction_model=AliasedQuantity)
    gold = json.dumps({"segments": [{"fields": {"Net/Weight~": 1}}]})

    result = ExtractionEvaluator(config)(response=gold, ground_truth=gold)

    assert result["field_counts"] == {"/segments/*/fields/Net~1Weight~0": [1, 0, 0]}


@pytest.mark.parametrize("prediction", ["invalid", {"nested": 1}, True])
def test_given_invalid_model_prediction_when_scored_then_directional_errors_preserved(prediction: object) -> None:
    """Count invalid present predictions as errors rather than dropping their fields."""
    config = EvaluationConfig(extraction_model=Quantity)
    gold = json.dumps({"segments": [{"fields": {"quantity": 1}}]})
    predicted = json.dumps({"segments": [{"fields": {"quantity": prediction}}]})

    result = ExtractionEvaluator(config)(response=predicted, ground_truth=gold)

    assert SchemaEvaluator(config)(response=predicted)["valid"] == 0
    assert (result["tp"], result["fp"], result["fn"]) == (0, 1, 1)


@pytest.mark.parametrize(("quantities", "fp", "fn"), [([1], 0, 1), ([1, 2, 2], 1, 0)])
def test_given_model_row_count_change_when_scored_then_multiplicity_preserved(quantities: list[int], fp: int, fn: int) -> None:
    """Retain directional counts for missing and extra selected row occurrences."""
    config = EvaluationConfig(extraction_model=Delivery)
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
        EvaluationConfig(extraction_model=Transforming)


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


@pytest.mark.parametrize("invalid_result", [None, 0, 1, 0.5, "true", {}, (True, True)])
def test_given_bad_comparator_return_when_invoked_then_contract_error_raised(invalid_result: object) -> None:
    """Validate runtime output even if caller annotations claim the correct type."""
    class BadResult(EvaluationModel):
        """Declare a comparator with a malformed return value."""

        value: int

        @staticmethod
        @field_comparator("value")
        def invalid_return(expected: int, actual: int) -> bool:
            """Return an invalid object to exercise the boundary check."""
            return cast(bool, invalid_result)

    config = EvaluationConfig(extraction_model=BadResult)
    gold = json.dumps({"segments": [{"fields": {"value": 1}}]})

    with pytest.raises(TypeError, match="must return bool"):
        ExtractionEvaluator(config)(response=gold, ground_truth=gold)