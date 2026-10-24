"""Verify extraction occurrence metrics against synthetic counterexamples."""

import json
from copy import deepcopy

import pytest

from extractly import EvaluationConfig, ExtractionEvaluator, FieldRule, SchemaEvaluator
from extractly.types import JsonObject


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


def test_given_reordered_segments_when_units_differ_then_original_unit_references_used(config: EvaluationConfig) -> None:
    """Resolve absolute unit pointers against each input's original row index."""
    config.fields["/segments/*/fields/quantity"] = FieldRule(kind="decimal", unit_field="/segments/*/fields/unit", unit_scales={"kg": "0.001", "MT": "1"})
    expected = {"segments": [
        {"category": "Events", "pages": [1, 1], "fields": {"unit": "kg", "quantity": 12000}},
        {"category": "Events", "pages": [2, 2], "fields": {"unit": "MT", "quantity": 12}},
    ]}

    result = ExtractionEvaluator(config)(response=json.dumps({"segments": expected["segments"][::-1]}), ground_truth=json.dumps(expected))

    assert result["exact_match"] == 1


@pytest.mark.parametrize("prediction", [True, "12", None, {"extra": 1}, "NaN"])
def test_given_wrong_decimal_type_when_compared_then_directional_errors(prediction: object) -> None:
    """Reject booleans, implicit numeric strings, containers, and absence."""
    config = EvaluationConfig(fields={"/quantity": FieldRule(kind="decimal")})

    result = ExtractionEvaluator(config)(response=json.dumps({"quantity": prediction}), ground_truth='{"quantity":12}')

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
    config = EvaluationConfig(fields={"/quantity": FieldRule(kind="decimal", unit_field="unit", unit_scales={"u": "1"})})
    number = 123456789012345678901234567890123456789

    result = ExtractionEvaluator(config)(response=json.dumps({"quantity": number + 1, "unit": "u"}), ground_truth=json.dumps({"quantity": number, "unit": "u"}))

    assert result["exact_match"] == 0
    assert result["tolerant"]["f1"] < 1


def test_given_explicit_text_policy_when_compared_then_whitespace_and_unicode_normalized() -> None:
    """Normalize only configured text rather than all string fields."""
    config = EvaluationConfig(fields={"/name": FieldRule(kind="text")})

    result = ExtractionEvaluator(config)(response='{"name":"  hello  world "}', ground_truth='{"name":"hello world"}')

    assert result["exact_match"] == 1


def test_given_invalid_gold_when_compared_then_not_prediction_failure() -> None:
    """Require trustworthy source annotations for typed comparison."""
    config = EvaluationConfig(fields={"/date": FieldRule(kind="date")})

    with pytest.raises(ValueError):
        ExtractionEvaluator(config)(response='{"date":"2026-01-01"}', ground_truth='{"date":"invalid"}')


def test_given_missing_required_key_when_schema_checked_then_violation(config: EvaluationConfig, record: JsonObject) -> None:
    """Report required omissions even when JSON parses."""
    value = json.loads(record["response"])
    del value["segments"][0]["fields"]["items"][0]["quantity"]

    result = SchemaEvaluator(config)(response=json.dumps(value))

    assert result["valid"] == 0
    assert "/segments/0/fields/items/0/quantity" in result["paths"]


def test_given_schema_format_when_invalid_then_violation() -> None:
    """Use jsonschema's format checker for configured standard formats."""
    config = EvaluationConfig(extraction_schema={"type": "object", "properties": {"date": {"type": "string", "format": "date"}}})

    result = SchemaEvaluator(config)(response='{"date":"not-a-date"}')

    assert result["valid"] == 0
