"""Verify the shared CU parser preserves segment-local text and detached values."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict

import pytest

from extractly import ParsedCUResult, ParsedCUSegment, parse_cu_result
from extractly.types import JsonObject


def test_given_evidence_when_parsed_then_only_segment_markdown_and_values_survive(raw: JsonObject) -> None:
    """Discard confidence, geometry, and spans without mutating CU input."""
    quantity = raw["contents"][0]["fields"]["items"]["valueArray"][0]["valueObject"]["quantity"]
    quantity.update(confidence=0.42, source="D(1,1,2,3,4)", spans=[{"offset": 10, "length": 5}])
    original = deepcopy(raw)

    parsed = parse_cu_result(raw)
    segment = asdict(parsed.segments[0])

    assert isinstance(parsed, ParsedCUResult)
    assert segment["content_path"] == "input1"
    assert segment["markdown"] == raw["contents"][0]["markdown"]
    assert segment["fields"]["items"][0] == {"eventId": "A", "quantity": 12000, "unit": "kg", "date": "2026-01-01"}
    assert not hasattr(parsed, "evidence")
    assert not hasattr(parsed, "response")
    assert raw == original


def test_given_omitted_field_when_parsed_then_not_invented(raw: JsonObject) -> None:
    """Leave missing properties absent rather than filling zero or null."""
    del raw["contents"][0]["fields"]["items"]["valueArray"][0]["valueObject"]["quantity"]

    segment = parse_cu_result(raw).segments[0]

    assert "quantity" not in segment.fields["items"][0]


def test_given_unrouted_segment_when_parsed_then_occurrence_retained() -> None:
    """Retain declarations without borrowing classifier Markdown."""
    raw = {"contents": [{"path": "input1", "markdown": "Unclassified content", "segments": [
        {"segmentId": "segment1", "category": "Other", "startPageNumber": 1, "endPageNumber": 1},
    ]}]}

    parsed = parse_cu_result(raw)

    assert parsed.segments == [ParsedCUSegment(content_path="input1/segment1", category="Other", pages=[1, 1], markdown="", fields={})]


def test_given_routed_segment_when_parsed_then_parent_not_duplicated(raw: JsonObject) -> None:
    """Use routed child text instead of the broader classifier context."""
    raw["contents"][0]["path"] = "input1/segment1"
    raw["contents"].insert(0, {"path": "input1", "markdown": "Other segment's source", "segments": [{"segmentId": "segment1"}]})

    parsed = parse_cu_result(raw)

    assert len(parsed.segments) == 1
    assert parsed.segments[0].markdown == raw["contents"][1]["markdown"]


def test_given_wrapped_result_when_parsed_then_values_and_encoding_survive(raw: JsonObject) -> None:
    """Accept saved successful operation envelopes and preserve span encoding."""
    raw["stringEncoding"] = "utf16"

    parsed = parse_cu_result({"status": "Succeeded", "result": raw})

    assert parsed.string_encoding == "utf16"
    assert parsed.segments[0].category == "Events"


def test_given_nested_values_when_parsed_then_types_and_occurrences_survive() -> None:
    """Retain escaped names, nulls, empty containers, zero, false, and duplicates."""
    raw = {"contents": [{"fields": {
        "a/b~c": {"valueNumber": 0}, "flag": {"valueBoolean": False}, "missing": None,
        "null_array": {"valueArray": None}, "null_object": {"valueObject": None},
        "empty_array": {"valueArray": []}, "empty_object": {"valueObject": {}},
        "rows": {"valueArray": [{"valueObject": {"name": {"valueString": "same"}}}] * 2},
    }}]}

    segment = parse_cu_result(raw).segments[0]

    assert segment.content_path == "content-0"
    assert segment.fields == {"a/b~c": 0, "flag": False, "missing": None, "null_array": None, "null_object": None, "empty_array": [], "empty_object": {}, "rows": [{"name": "same"}, {"name": "same"}]}


@pytest.mark.parametrize("value_key", ["value", "valueJson"])
def test_given_mutable_value_when_parsed_result_edited_then_raw_unchanged(value_key: str) -> None:
    """Detach generic JSON containers as well as typed arrays and objects."""
    raw = {"contents": [{"fields": {"details": {value_key: {"names": ["original"]}}}}]}
    original = deepcopy(raw)

    parsed = parse_cu_result(raw)
    parsed.segments[0].fields["details"]["names"].append("corrected")

    assert raw == original


def test_given_multiple_contents_when_parsed_then_markdown_is_segment_local(raw: JsonObject) -> None:
    """Associate Markdown and normalized fields without a separate evidence map."""
    second = deepcopy(raw["contents"][0])
    second.update(path="input2", markdown="Second segment only")
    raw["contents"].append(second)

    parsed = parse_cu_result(raw)

    assert all(isinstance(segment, ParsedCUSegment) for segment in parsed.segments)
    assert [segment.content_path for segment in parsed.segments] == ["input1", "input2"]
    assert [segment.markdown for segment in parsed.segments] == [raw["contents"][0]["markdown"], "Second segment only"]
    assert all(segment.fields["items"][0]["quantity"] == 12000 for segment in parsed.segments)


def test_given_parsed_segment_when_serialized_then_json_keys_are_unchanged(raw: JsonObject) -> None:
    """Retain canonical segment dictionaries for runner serialization."""
    segment = parse_cu_result(raw).segments[0]

    serialized = asdict(segment)

    assert serialized == {"content_path": "input1", "category": "Events", "pages": [1, 1], "markdown": raw["contents"][0]["markdown"], "fields": {"items": [{"eventId": "A", "quantity": 12000, "unit": "kg", "date": "2026-01-01"}, {"eventId": "B", "quantity": 8000, "unit": "kg", "date": "2026-01-02"}]}}


def test_given_frozen_segment_when_attribute_reassigned_then_rejected(raw: JsonObject) -> None:
    """Keep segment attribute references fixed after parsing."""
    segment = parse_cu_result(raw).segments[0]

    with pytest.raises(FrozenInstanceError):
        segment.__setattr__("markdown", "replacement")


@pytest.mark.parametrize("raw", [
    {"contents": []}, {"contents": [None]}, {"contents": [{"fields": []}]},
    {"contents": [{"fields": {"bad": {"valueArray": "invalid"}}}]},
    {"contents": [{"fields": {"bad": {"valueUnknown": 1}}}]},
    {"contents": [{"fields": {"bad": {"valueNumber": float("nan")}}}]},
    {"contents": [{"fields": {"bad": {"valueNumber": float("inf")}}}]},
    {"contents": [{"markdown": 12}]}, {"contents": [{"segments": [{"category": "Other"}]}]},
    {"contents": [{"path": "same"}, {"path": "same"}]},
    {"status": "Failed", "result": {"contents": [{}]}},
])
def test_given_malformed_cu_when_parsed_then_rejected(raw: JsonObject) -> None:
    """Reject invalid saved results through the single parsing entry point."""
    with pytest.raises(ValueError):
        parse_cu_result(raw)


@pytest.mark.parametrize("raw", [None, [], "not-cu"])
def test_given_unsupported_input_when_parsed_then_type_error(raw: JsonObject) -> None:
    """Require a mapping rather than paths or serialized JSON."""
    with pytest.raises(TypeError, match="raw CU dictionary"):
        parse_cu_result(raw)