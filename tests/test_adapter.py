"""Verify raw values and source associations survive adaptation."""

import json

import pytest

from extractly import CUAdapter, EvaluationConfig, EvaluationModel
from extractly.types import JsonObject


def test_given_evidence_when_prepared_then_source_metadata_survives(config: EvaluationConfig, raw: JsonObject) -> None:
    """Keep confidence, geometry, spans, and Markdown separate from values."""
    quantity = raw["contents"][0]["fields"]["items"]["valueArray"][0]["valueObject"]["quantity"]
    quantity.update(confidence=0.42, source="D(1,1,2,3,4)", spans=[{"offset": 10, "length": 5}])

    record = CUAdapter(config).prepare(raw, "source")
    evidence = record["evidence"]["/segments/0/fields/items/0/quantity"]

    assert evidence["confidence"] == 0.42
    assert evidence["source"] == quantity["source"]
    assert evidence["spans"] == quantity["spans"]
    assert record["markdown"]["input1"] == raw["contents"][0]["markdown"]
    assert json.loads(record["response"])["segments"][0]["fields"]["items"][0]["quantity"] == 12000


def test_given_required_omission_when_prepared_then_explicit_absent_slot(config: EvaluationConfig, raw: JsonObject) -> None:
    """Expose required absence to the field grader without inventing zero."""
    del raw["contents"][0]["fields"]["items"]["valueArray"][0]["valueObject"]["quantity"]
    adapter = CUAdapter(config)

    record = adapter.prepare(raw, "source")
    rows = adapter.field_rows(record)
    quantity = next(row for row in rows if row["field_path"] == "/segments/0/fields/items/0/quantity")

    assert json.loads(quantity["response"]) == {"presence": "absent"}
    assert "quantity" not in json.loads(record["response"])["segments"][0]["fields"]["items"][0]


def test_given_field_grading_when_generated_then_confidence_does_not_anchor_prompt(config: EvaluationConfig, raw: JsonObject) -> None:
    """Provide row identity but exclude model confidence from judge inputs."""
    adapter = CUAdapter(config)
    prepared = adapter.prepare(raw, "source")
    prepared["evidence"]["/segments/0/fields/items/0/quantity"]["confidence"] = 1.0

    rows = adapter.field_rows(prepared)
    quantity = next(row for row in rows if row["field_path"] == "/segments/0/fields/items/0/quantity")

    assert "confidence" not in json.dumps(rows)
    assert all("/fields/" in row["field_path"] for row in rows)
    assert json.loads(quantity["row_context"])["eventId"] == "A"
    assert json.loads(quantity["contract"]) == config.model_dump(mode="json", include={"extraction_schema", "tables"})


def test_given_unrouted_segment_when_prepared_then_occurrence_retained(config: EvaluationConfig) -> None:
    """Retain classifier instances whose analyzer did not produce fields."""
    raw = {"contents": [{
        "path": "input1", "markdown": "Unclassified content",
        "segments": [{"segmentId": "segment1", "category": "Other", "startPageNumber": 1, "endPageNumber": 1}],
    }]}

    prepared = CUAdapter(config).prepare(raw, "source")

    assert json.loads(prepared["response"])["segments"] == [{"category": "Other", "pages": [1, 1], "fields": {}}]


def test_given_routed_segment_when_prepared_then_not_duplicated(config: EvaluationConfig, raw: JsonObject) -> None:
    """Resolve child content by its declared segment path."""
    child = raw["contents"][0]
    child["path"] = "input1/segment1"
    raw["contents"].insert(0, {"path": "input1", "segments": [{"segmentId": "segment1"}]})

    prepared = CUAdapter(config).prepare(raw, "source")

    assert len(json.loads(prepared["response"])["segments"]) == 1


def test_given_nested_fields_when_adapted_then_source_names_survive(config: EvaluationConfig, raw: JsonObject) -> None:
    """Preserve independent source field names in values and evidence."""

    record = CUAdapter(config).prepare(raw, "source")
    item = json.loads(record["response"])["segments"][0]["fields"]["items"][0]

    assert {"unit", "eventId"} <= item.keys()
    assert "/segments/0/fields/items/0/unit" in record["evidence"]
    assert "/segments/0/fields/items/0/eventId" in record["evidence"]


@pytest.mark.parametrize("limit", ["context", "fields"])
def test_given_excess_budget_when_prepared_then_not_truncated(limit: str, config: EvaluationConfig, raw: JsonObject) -> None:
    """Reject oversized evidence or grading work instead of hiding omissions."""
    adapter = CUAdapter(config)
    if limit == "context":
        config.max_context_characters = 1
        with pytest.raises(ValueError, match="never truncate"):
            adapter.prepare(raw, "source")
    else:
        with pytest.raises(ValueError, match="field-grading budget"):
            adapter.field_rows(adapter.prepare(raw, "source"), max_fields_per_document=1)


@pytest.mark.parametrize("raw", [
    {"contents": []},
    {"contents": [None]},
    {"contents": [{"fields": []}]},
    {"contents": [{"fields": {"x": {"valueArray": "not-an-array"}}}]},
    {"contents": [{"markdown": 12}]},
    {"status": "Failed", "result": {"contents": [{}]}},
])
def test_given_malformed_cu_when_prepared_then_rejected(raw: JsonObject) -> None:
    """Surface malformed data instead of returning empty successful extraction."""
    with pytest.raises(ValueError):
        CUAdapter(EvaluationConfig()).prepare(raw, "source")


def test_given_wrapped_result_when_prepared_then_values_and_encoding_survive(raw: JsonObject) -> None:
    """Accept saved long-running CU operation envelopes."""
    wrapped = {"status": "Succeeded", "result": raw}

    result = CUAdapter(EvaluationConfig()).prepare(wrapped, "source")

    assert result["string_encoding"] == "codePoint"
    assert json.loads(result["response"])["segments"][0]["category"] == "Events"


def test_given_escaped_field_names_when_adapted_then_pointer_is_unambiguous() -> None:
    """Escape slash and tilde tokens in source field names."""
    raw = {"contents": [{"fields": {"a/b~c": {"valueNumber": 0}, "flag": {"valueBoolean": False}}}]}

    record = CUAdapter(EvaluationConfig()).prepare(raw, "source")

    assert record["evidence"]["/segments/0/fields/a~1b~0c"]["content_path"] == "content-0"
    assert json.loads(record["response"])["segments"][0]["fields"] == {"a/b~c": 0, "flag": False}


@pytest.mark.parametrize("missing_quantity", [False, True])
def test_given_selected_fields_when_grading_then_only_selected_slots_consume_budget(missing_quantity: bool, config: EvaluationConfig, raw: JsonObject, quantity_schema: type[EvaluationModel]) -> None:
    """Filter unrelated emitted/required fields without losing source context."""
    config.extraction_schema = quantity_schema
    first = raw["contents"][0]["fields"]["items"]["valueArray"][0]["valueObject"]
    del first["date"]
    if missing_quantity:
        del first["quantity"]
    adapter = CUAdapter(config)
    prepared = adapter.prepare(raw, "source")

    rows = adapter.field_rows(prepared, max_fields_per_document=2)
    quantity = next(row for row in rows if row["field_path"] == "/segments/0/fields/items/0/quantity")

    assert {row["field_path"] for row in rows} == {"/segments/0/fields/items/0/quantity", "/segments/0/fields/items/1/quantity"}
    assert json.loads(quantity["contract"]) == config.model_dump(mode="json", include={"extraction_schema", "tables"})
    assert json.loads(quantity["row_context"])["eventId"] == "A"
    assert json.loads(quantity["response"]) == ({"presence": "absent"} if missing_quantity else {"value": 12000})
