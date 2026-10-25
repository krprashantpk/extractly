"""Verify extraction occurrence metrics against synthetic counterexamples."""

import json
import unicodedata
from copy import deepcopy
from datetime import date
from decimal import Decimal

import pytest
from pydantic import Field

from extractly import (
    Comparison,
    EvaluationConfig,
    EvaluationModel,
    ExtractionEvaluator,
    SchemaEvaluator,
    field_comparator,
)
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


def test_given_identical_when_compared_then_exact_match(config: EvaluationConfig, record: JsonObject) -> None:
    """Keep field, row, and document agreement distinct."""
    result = ExtractionEvaluator(config)(response=record["response"], ground_truth=record["response"])

    assert result["f1"] == result["exact_match"] == result["row_recall"] == 1


def test_given_swapped_quantities_when_totals_balance_then_two_wrong_values(config: EvaluationConfig, record: JsonObject) -> None:
    """Do not accept balancing totals as correct row associations."""
    predicted = json.loads(record["response"])
    rows = predicted["segments"][0]["fields"]["items"]
    rows[0]["quantity"], rows[1]["quantity"] = rows[1]["quantity"], rows[0]["quantity"]

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=record["response"])

    assert result["field_counts"]["/segments/*/fields/items/*/quantity"] == [0, 2, 2]
    assert result["exact_match"] == 0
    assert result["row_recall"] == 1


@pytest.mark.parametrize(("operation", "fp", "fn"), [("delete", 0, 1), ("duplicate", 1, 0)])
def test_given_missing_or_extra_row_when_compared_then_multiplicity_preserved(operation: str, fp: int, fn: int, config: EvaluationConfig, record: JsonObject) -> None:
    """Keep duplicate and missing row occurrence counts."""
    predicted = json.loads(record["response"])
    rows = predicted["segments"][0]["fields"]["items"]
    if operation == "delete":
        rows.pop()
    else:
        rows.append(deepcopy(rows[0]))

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=record["response"])

    assert result["field_counts"]["/segments/*/fields/items/*/quantity"][1:] == [fp, fn]


def test_given_tolerance_when_compared_then_strict_exact_match_stays_false(config: EvaluationConfig, record: JsonObject) -> None:
    """Do not let tolerant scores overwrite strict normalized comparison."""
    predicted = json.loads(record["response"])
    predicted["segments"][0]["fields"]["items"][0]["quantity"] += 1

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=record["response"])

    assert result["tolerant"]["f1"] == 1
    assert result["exact_match"] == 0


def test_given_reordered_rows_when_keys_match_then_not_false_error(config: EvaluationConfig, record: JsonObject) -> None:
    """Match independent identities rather than row position."""
    predicted = json.loads(record["response"])
    predicted["segments"][0]["fields"]["items"].reverse()

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=record["response"])

    assert result["exact_match"] == 1


def test_given_reordered_segments_when_units_differ_then_original_values_align(config: EvaluationConfig) -> None:
    """Resolve values against each input's keyed original segment."""
    config.extraction_schema = Quantity
    expected = {"segments": [
        {"category": "Events", "pages": [1, 1], "fields": {"unit": "kg", "quantity": 12000}},
        {"category": "Events", "pages": [2, 2], "fields": {"unit": "MT", "quantity": 12}},
    ]}

    result = ExtractionEvaluator(config)(response=json.dumps({"segments": expected["segments"][::-1]}), ground_truth=json.dumps(expected))

    assert result["exact_match"] == 1


@pytest.mark.parametrize("prediction", [True, "12", None, {"extra": 1}, "NaN"])
def test_given_wrong_decimal_type_when_compared_then_directional_errors(prediction: object) -> None:
    """Reject booleans, implicit numeric strings, containers, and absence."""
    config = EvaluationConfig(extraction_schema=Quantity)

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

    config = EvaluationConfig(extraction_schema=DecimalQuantity)
    number = 123456789012345678901234567890123456789

    result = ExtractionEvaluator(config)(response=canonical({"quantity": number + 1}), ground_truth=canonical({"quantity": number}))

    assert result["exact_match"] == 0
    assert result["tolerant"]["f1"] < 1


def test_given_explicit_text_policy_when_compared_then_whitespace_and_unicode_normalized() -> None:
    """Normalize only configured text rather than all string fields."""
    class Named(EvaluationModel):
        """Keep domain normalization in a registered caller function."""

        name: str

        @staticmethod
        @field_comparator("name")
        def normalized_text(expected: str, actual: str) -> Comparison:
            """Compare whitespace and canonical Unicode forms explicitly."""
            equal = " ".join(unicodedata.normalize("NFC", expected).split()) == " ".join(unicodedata.normalize("NFC", actual).split())
            return Comparison(equal, equal)

    config = EvaluationConfig(extraction_schema=Named)

    result = ExtractionEvaluator(config)(response=canonical({"name": "  hello  world "}), ground_truth=canonical({"name": "hello world"}))

    assert result["exact_match"] == 1


def test_given_invalid_gold_when_compared_then_not_prediction_failure() -> None:
    """Require trustworthy source annotations for typed comparison."""
    class Dated(EvaluationModel):
        """Require valid ISO calendar dates."""

        date: date

    config = EvaluationConfig(extraction_schema=Dated)

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

    config = EvaluationConfig(extraction_schema=Dated)

    result = SchemaEvaluator(config)(response=canonical({"date": "not-a-date"}))

    assert result["valid"] == 0


def test_given_excluded_errors_when_selected_fields_match_then_perfect_selected_scores() -> None:
    """Exclude unrelated values, containers, and metadata from selected metrics."""
    config = EvaluationConfig(extraction_schema=Quantity)
    expected = {"segments": [{"category": "Detail", "pages": [1, 1], "fields": {"quantity": 10, "date": "invalid", "name": "expected", "unused": []}}]}
    predicted = {"segments": [{"category": "Other", "pages": [8, 9], "fields": {"quantity": 10, "date": 42, "name": {"extra": "wrong"}}}]}

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=json.dumps(expected))

    assert result["f1"] == result["exact_match"] == 1
    assert result["field_counts"] == {"/segments/*/fields/quantity": [1, 0, 0]}


@pytest.mark.parametrize(("prediction", "fp", "fn"), [({"quantity": 11, "garbage": "ignored"}, 1, 1), ({"garbage": "ignored"}, 0, 1)])
def test_given_selected_error_when_compared_then_directional_counts_remain(prediction: JsonObject, fp: int, fn: int) -> None:
    """Keep wrong and missing selected values visible without counting extras."""
    config = EvaluationConfig(extraction_schema=Quantity)

    result = ExtractionEvaluator(config)(response=canonical(prediction), ground_truth=canonical({"quantity": 10, "unused": "gold"}))

    assert (result["tp"], result["fp"], result["fn"]) == (0, fp, fn)
    assert result["exact_match"] == 0
    assert set(result["field_counts"]) == {"/segments/*/fields/quantity"}


def test_given_selected_quantities_when_rows_reordered_then_excluded_keys_still_align(config: EvaluationConfig, record: JsonObject) -> None:
    """Use independent row keys without scoring those excluded identity fields."""
    config.extraction_schema = Items
    predicted = json.loads(record["response"])
    predicted["segments"][0]["fields"]["items"].reverse()

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=record["response"])

    assert result["f1"] == result["exact_match"] == 1
    assert set(result["field_counts"]) == {"/segments/*/fields/items/*/quantity"}


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
        def in_tonnes(expected: Measurement, actual: Measurement) -> Comparison:
            """Convert declared units in caller-owned policy, not the engine."""
            scales = {"kg": Decimal("0.001"), "MT": Decimal("1")}
            equal = expected.value * scales[expected.unit] == actual.value * scales[actual.unit]
            return Comparison(equal, equal)

    config = EvaluationConfig(extraction_schema=MeasuredQuantity)

    result = ExtractionEvaluator(config)(response=canonical({"quantity": {"value": 1000, "unit": "kg"}}), ground_truth=canonical({"quantity": {"value": 1, "unit": "MT"}}))

    assert result["f1"] == result["exact_match"] == 1
    assert result["field_counts"] == {"/segments/*/fields/quantity": [1, 0, 0]}


def test_given_excluded_required_field_when_missing_then_no_slot_violation() -> None:
    """Undeclared fields cannot introduce a separate requiredness policy."""
    config = EvaluationConfig(extraction_schema=Quantity)

    result = SchemaEvaluator(config)(response=canonical({"quantity": 10}))

    assert result["valid"] == 1


def test_given_selected_required_field_when_missing_then_slot_violation() -> None:
    """Keep selected required omissions visible to schema quality gates."""
    config = EvaluationConfig(extraction_schema=Quantity)

    result = SchemaEvaluator(config)(response=canonical({"unused": 10}))

    assert result["paths"] == ["/segments/0/fields/quantity"]


def test_given_declared_required_fields_when_missing_then_contract_validated() -> None:
    """Use the same model for requiredness and scored field selection."""
    class NamedQuantity(Quantity):
        """Add a selected required name."""

        name: str

    config = EvaluationConfig(extraction_schema=NamedQuantity)

    result = SchemaEvaluator(config)(response=canonical({"quantity": 10}))

    assert result["valid"] == 0
    assert config.selects_path("/segments/0/fields/name")


@pytest.mark.parametrize(("operation", "fp", "fn"), [("delete", 0, 1), ("duplicate", 1, 0)])
def test_given_selected_values_when_row_count_changes_then_occurrences_still_count(operation: str, fp: int, fn: int, config: EvaluationConfig, record: JsonObject) -> None:
    """Do not let selection hide missing or duplicate rows with selected values."""
    config.extraction_schema = Items
    predicted = json.loads(record["response"])
    rows = predicted["segments"][0]["fields"]["items"]
    if operation == "delete":
        rows.pop()
    else:
        rows.append(deepcopy(rows[0]))

    result = ExtractionEvaluator(config)(response=json.dumps(predicted), ground_truth=record["response"])

    assert result["field_counts"]["/segments/*/fields/items/*/quantity"][1:] == [fp, fn]
    assert result["exact_match"] == 0


def test_given_declared_list_when_other_row_added_then_extra_occurrence_scored() -> None:
    """Apply the model to every occurrence without a parallel path allowlist."""
    config = EvaluationConfig(extraction_schema=Items)

    result = ExtractionEvaluator(config)(response=canonical({"items": [{"quantity": 10}, {"quantity": 99}]}), ground_truth=canonical({"items": [{"quantity": 10}]}))

    assert result["exact_match"] == 0
    assert result["field_counts"] == {"/segments/*/fields/items/*/quantity": [1, 1, 0]}


def test_given_selected_container_and_escaped_field_when_compared_then_descendants_included() -> None:
    """Use pointer escaping and include descendants without similar prefixes."""
    class EscapedItems(Items):
        """Select an aliased source field alongside nested item quantities."""

        escaped: bool = Field(alias="a/b~c")

    config = EvaluationConfig(extraction_schema=EscapedItems)

    result = ExtractionEvaluator(config)(response=canonical({"a/b~c": False, "items": [{"quantity": 0}], "itemsExtra": 99}), ground_truth=canonical({"a/b~c": False, "items": [{"quantity": 0}], "itemsExtra": 1}))

    assert result["f1"] == result["exact_match"] == 1
    assert set(result["field_counts"]) == {"/segments/*/fields/a~1b~0c", "/segments/*/fields/items/*/quantity"}
