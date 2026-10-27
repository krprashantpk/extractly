"""Verify extraction occurrence metrics against synthetic counterexamples."""

import json
import unicodedata
from copy import deepcopy
from datetime import date
from decimal import Decimal

import pytest
from pydantic import Field

from extractly import (
    EvaluationConfig,
    EvaluationModel,
    ExtractionEvaluator,
    SchemaEvaluator,
    field_comparator,
)
from extractly.evaluators import ratios
from extractly.types import JsonObject


class Quantity(EvaluationModel):
    """Select a strict integer quantity without unrelated source fields."""

    quantity: int = Field(strict=True)


class Items(EvaluationModel):
    """Select quantities in every existing item occurrence."""

    items: list[Quantity]


def canonical(fields: JsonObject) -> str:
    """Encode segment fields in the public canonical document envelope."""
    return json.dumps({"segments": [{"fields": fields}]})


@pytest.mark.parametrize(("tp", "fp", "fn", "expected"), [
    (1, 2, 3, {"precision": 0.333, "recall": 0.25, "f1": 0.286}),
    (187, 1, 1, {"precision": 0.995, "recall": 0.995, "f1": 0.995}),
    (0, 0, 0, {"precision": None, "recall": None, "f1": None}),
    (0, 1, 0, {"precision": 0.0, "recall": None, "f1": 0.0}),
    (0, 0, 1, {"precision": None, "recall": 0.0, "f1": 0.0}),
])
def test_given_occurrence_counts_when_ratios_calculated_then_rounded_and_undefined_preserved(tp: int, fp: int, fn: int, expected: dict[str, float | None]) -> None:
    """Round completed ratios without converting undefined scores to numbers."""
    result = ratios(tp, fp, fn)

    assert result == expected


def test_given_identical_when_compared_then_exact_match(config: EvaluationConfig, record: JsonObject) -> None:
    """Keep field, row, and document agreement distinct."""
    result = ExtractionEvaluator(config)(response=record["response"], ground_truth=record["response"])

    assert result["f1"] == result["exact_match"] == result["table_row_recall"] == result["table_count_recall"] == 1
    assert result["table_row_tp"] == 2
    assert result["table_count_tp"] == 1
    assert not any(key.startswith(("row_", "instance_")) for key in result)


def test_given_swapped_quantities_when_totals_balance_then_two_wrong_values(config: EvaluationConfig, record: JsonObject) -> None:
    """Do not accept balancing totals as correct row associations."""
    predicted = json.loads(record["response"])
    rows = predicted["segments"][0]["fields"]["items"]
    rows[0]["quantity"], rows[1]["quantity"] = rows[1]["quantity"], rows[0]["quantity"]

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=record["response"])

    assert result["field_counts"]["/segments/*/fields/items/*/quantity"] == [0, 2, 2]
    assert result["exact_match"] == 0
    assert result["table_row_recall"] == 1


@pytest.mark.parametrize(("operation", "fp", "fn"), [("delete", 0, 1), ("duplicate", 1, 0)])
@pytest.mark.parametrize("quantity_only", [False, True])
def test_given_missing_or_extra_row_when_compared_then_multiplicity_preserved(operation: str, fp: int, fn: int, config: EvaluationConfig, record: JsonObject, quantity_only: bool) -> None:
    """Keep duplicate and missing row counts with full or quantity-only selection."""
    if quantity_only:
        config.extraction_model = Items
    predicted = json.loads(record["response"])
    rows = predicted["segments"][0]["fields"]["items"]
    if operation == "delete":
        rows.pop()
    else:
        rows.append(deepcopy(rows[0]))

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=record["response"])

    assert result["field_counts"]["/segments/*/fields/items/*/quantity"][1:] == [fp, fn]
    assert result["table_row_fp"] == fp
    assert result["table_row_fn"] == fn
    assert result["exact_match"] == 0


def test_given_allowed_variation_when_compared_then_document_matches(config: EvaluationConfig, record: JsonObject) -> None:
    """Treat an allowed quantity difference as an ordinary accepted match."""
    predicted = json.loads(record["response"])
    predicted["segments"][0]["fields"]["items"][0]["quantity"] += 1

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=record["response"])

    assert result["f1"] == result["exact_match"] == 1


@pytest.mark.parametrize("quantity_only", [False, True])
def test_given_reordered_rows_when_compared_then_positional_errors(config: EvaluationConfig, record: JsonObject, quantity_only: bool) -> None:
    """Reject reordered values even when source identities are unchanged."""
    if quantity_only:
        config.extraction_model = Items
    predicted = json.loads(record["response"])
    predicted["segments"][0]["fields"]["items"].reverse()

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=record["response"])

    assert result["f1"] < 1
    assert result["exact_match"] == 0
    assert result["table_row_tp"] == 2
    assert result["table_row_fp"] == result["table_row_fn"] == 0
    assert result["field_counts"]["/segments/*/fields/items/*/quantity"] == [0, 2, 2]
    if quantity_only:
        assert set(result["field_counts"]) == {"/segments/*/fields/items/*/quantity"}


def test_given_reordered_segments_when_compared_then_positional_errors(config: EvaluationConfig) -> None:
    """Reject reordered segments rather than matching their source identities."""
    config.extraction_model = Quantity
    expected = {"segments": [
        {"category": "Events", "pages": [1, 1], "fields": {"unit": "kg", "quantity": 12000}},
        {"category": "Events", "pages": [2, 2], "fields": {"unit": "MT", "quantity": 12}},
    ]}

    result = ExtractionEvaluator(config)(response=json.dumps({"segments": expected["segments"][::-1]}), ground_truth=json.dumps(expected))

    assert result["exact_match"] == 0
    assert result["table_count_tp"] == 2
    assert result["field_counts"]["/segments/*/fields/quantity"] == [0, 2, 2]


@pytest.mark.parametrize(("operation", "fp", "fn"), [("delete", 0, 1), ("duplicate", 1, 0)])
def test_given_missing_or_extra_table_when_compared_then_table_count_errors(operation: str, fp: int, fn: int, config: EvaluationConfig, record: JsonObject) -> None:
    """Count absent and additional tables independently of their row values."""
    predicted = json.loads(record["response"])
    if operation == "delete":
        predicted["segments"].pop()
    else:
        predicted["segments"].append(deepcopy(predicted["segments"][0]))

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=record["response"])

    assert result["table_count_fp"] == fp
    assert result["table_count_fn"] == fn
    assert result["table_row_fp"] == 2 * fp
    assert result["table_row_fn"] == 2 * fn


@pytest.mark.parametrize("prediction", [True, "12", None, {"extra": 1}, "NaN"])
def test_given_wrong_decimal_type_when_compared_then_directional_errors(prediction: object) -> None:
    """Reject booleans, implicit numeric strings, containers, and absence."""
    config = EvaluationConfig(extraction_model=Quantity)

    result = ExtractionEvaluator(config)(response=canonical({"quantity": prediction}), ground_truth=canonical({"quantity": 12}))

    assert result["tp"] == 0
    assert result["fn"] == 1
    assert result["fp"] == (0 if prediction is None else 1)


def test_given_no_positive_values_when_compared_then_f1_undefined() -> None:
    """Do not treat all-empty extraction as perfect positive recall."""
    result = ExtractionEvaluator(EvaluationConfig())(response='{"value":null}', ground_truth='{"value":null}')

    assert result["f1"] is None


def test_given_missing_empty_container_when_compared_then_document_not_exact() -> None:
    """Preserve structural absence even when it contributes no leaf values."""
    result = ExtractionEvaluator(EvaluationConfig())(response="{}", ground_truth='{"items":[]}')

    assert result["exact_match"] == 0


def test_given_large_decimal_when_converted_then_rounding_does_not_hide_error() -> None:
    """Preserve a one-unit difference beyond the default Decimal precision."""
    class DecimalQuantity(EvaluationModel):
        """Validate exact decimals without implicit arithmetic rounding."""

        quantity: Decimal

    config = EvaluationConfig(extraction_model=DecimalQuantity)
    number = 123456789012345678901234567890123456789

    result = ExtractionEvaluator(config)(response=canonical({"quantity": number + 1}), ground_truth=canonical({"quantity": number}))

    assert result["exact_match"] == 0
    assert result["f1"] < 1


def test_given_explicit_text_policy_when_compared_then_whitespace_and_unicode_normalized() -> None:
    """Normalize only configured text rather than all string fields."""
    class Named(EvaluationModel):
        """Keep domain normalization in a registered caller function."""

        name: str

        @staticmethod
        @field_comparator("name")
        def normalized_text(expected: str, actual: str) -> bool:
            """Compare whitespace and canonical Unicode forms explicitly."""
            equal = " ".join(unicodedata.normalize("NFC", expected).split()) == " ".join(unicodedata.normalize("NFC", actual).split())
            return equal

    config = EvaluationConfig(extraction_model=Named)

    result = ExtractionEvaluator(config)(response=canonical({"name": "  hello  world "}), ground_truth=canonical({"name": "hello world"}))

    assert result["exact_match"] == 1


def test_given_invalid_gold_when_compared_then_not_prediction_failure() -> None:
    """Require trustworthy source annotations for typed comparison."""
    class Dated(EvaluationModel):
        """Require valid ISO calendar dates."""

        date: date

    config = EvaluationConfig(extraction_model=Dated)

    with pytest.raises(ValueError):
        ExtractionEvaluator(config)(response=canonical({"date": "2026-01-01"}), ground_truth=canonical({"date": "invalid"}))


def test_given_missing_required_key_when_schema_checked_then_violation(config: EvaluationConfig, record: JsonObject) -> None:
    """Report required omissions even when JSON parses."""
    value = json.loads(record["response"])
    del value["segments"][0]["fields"]["items"][0]["quantity"]

    result = SchemaEvaluator(config)(response=json.dumps(value))

    assert result["valid"] == 0
    assert "/segments/0/fields/items/0/quantity" in result["paths"]


def test_given_schema_format_when_invalid_then_violation() -> None:
    """Use the extraction model's date type for calendar validation."""
    class Dated(EvaluationModel):
        """Require valid calendar dates."""

        date: date

    config = EvaluationConfig(extraction_model=Dated)

    result = SchemaEvaluator(config)(response=canonical({"date": "not-a-date"}))

    assert result["valid"] == 0


def test_given_excluded_errors_when_selected_fields_match_then_perfect_selected_scores() -> None:
    """Exclude unrelated values, containers, and metadata from selected metrics."""
    config = EvaluationConfig(extraction_model=Quantity)
    expected = {"segments": [{"category": "Detail", "pages": [1, 1], "fields": {"quantity": 10, "date": "invalid", "name": "expected", "unused": []}}]}
    predicted = {"segments": [{"category": "Other", "pages": [8, 9], "fields": {"quantity": 10, "date": 42, "name": {"extra": "wrong"}}}]}

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=json.dumps(expected))

    assert result["f1"] == result["exact_match"] == 1
    assert result["field_counts"] == {"/segments/*/fields/quantity": [1, 0, 0]}


@pytest.mark.parametrize(("prediction", "fp", "fn"), [({"quantity": 11, "garbage": "ignored"}, 1, 1), ({"garbage": "ignored"}, 0, 1)])
def test_given_selected_error_when_compared_then_directional_counts_remain(prediction: JsonObject, fp: int, fn: int) -> None:
    """Keep wrong and missing selected values visible without counting extras."""
    config = EvaluationConfig(extraction_model=Quantity)

    result = ExtractionEvaluator(config)(response=canonical(prediction), ground_truth=canonical({"quantity": 10, "unused": "gold"}))

    assert (result["tp"], result["fp"], result["fn"]) == (0, fp, fn)
    assert result["exact_match"] == 0
    assert set(result["field_counts"]) == {"/segments/*/fields/quantity"}


def test_given_compound_quantity_when_compared_then_domain_unit_conversion_works() -> None:
    """Declare unit-dependent values together for a two-value comparator."""
    class Measurement(EvaluationModel):
        """Represent the value and unit consumed by a compound comparator."""

        value: Decimal
        unit: str

    class MeasuredQuantity(EvaluationModel):
        """Score the complete measurement as one field."""

        quantity: Measurement

        @staticmethod
        @field_comparator("quantity")
        def in_tonnes(expected: Measurement, actual: Measurement) -> bool:
            """Convert declared units in caller-owned policy, not the engine."""
            scales = {"kg": Decimal("0.001"), "MT": Decimal("1")}
            equal = expected.value * scales[expected.unit] == actual.value * scales[actual.unit]
            return equal

    config = EvaluationConfig(extraction_model=MeasuredQuantity)

    result = ExtractionEvaluator(config)(response=canonical({"quantity": {"value": 1000, "unit": "kg"}}), ground_truth=canonical({"quantity": {"value": 1, "unit": "MT"}}))

    assert result["f1"] == result["exact_match"] == 1
    assert result["field_counts"] == {"/segments/*/fields/quantity": [1, 0, 0]}


@pytest.mark.parametrize(("fields", "paths"), [({"quantity": 10}, []), ({"unused": 10}, ["/segments/0/fields/quantity"])])
def test_given_selected_required_field_when_schema_checked_then_only_declared_slots_validated(fields: JsonObject, paths: list[str]) -> None:
    """Validate selected required slots without inventing undeclared requirements."""
    config = EvaluationConfig(extraction_model=Quantity)

    result = SchemaEvaluator(config)(response=canonical(fields))

    assert result["paths"] == paths
    assert result["valid"] == int(not paths)


def test_given_declared_required_fields_when_missing_then_contract_validated() -> None:
    """Use the same model for requiredness and scored field selection."""
    class NamedQuantity(Quantity):
        """Add a selected required name."""

        name: str

    config = EvaluationConfig(extraction_model=NamedQuantity)

    result = SchemaEvaluator(config)(response=canonical({"quantity": 10}))

    assert result["valid"] == 0
    assert config.selects_path("/segments/0/fields/name")


def test_given_declared_list_when_other_row_added_then_extra_occurrence_scored() -> None:
    """Apply the model to every occurrence without a parallel path allowlist."""
    config = EvaluationConfig(extraction_model=Items)

    result = ExtractionEvaluator(config)(response=canonical({"items": [{"quantity": 10}, {"quantity": 99}]}), ground_truth=canonical({"items": [{"quantity": 10}]}))

    assert result["exact_match"] == 0
    assert result["field_counts"] == {"/segments/*/fields/items/*/quantity": [1, 1, 0]}


def test_given_selected_container_and_escaped_field_when_compared_then_descendants_included() -> None:
    """Use pointer escaping and include descendants without similar prefixes."""
    class EscapedItems(Items):
        """Select an aliased source field alongside nested item quantities."""

        escaped: bool = Field(alias="a/b~c")

    config = EvaluationConfig(extraction_model=EscapedItems)

    result = ExtractionEvaluator(config)(response=canonical({"a/b~c": False, "items": [{"quantity": 0}], "itemsExtra": 99}), ground_truth=canonical({"a/b~c": False, "items": [{"quantity": 0}], "itemsExtra": 1}))

    assert result["f1"] == result["exact_match"] == 1
    assert set(result["field_counts"]) == {"/segments/*/fields/a~1b~0c", "/segments/*/fields/items/*/quantity"}
