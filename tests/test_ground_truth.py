"""Verify CU-derived annotation drafts share the adapter's canonical contract."""

import json
from copy import deepcopy

import pytest

from extractly import CUAdapter, EvaluationConfig, build_ground_truth
from extractly.types import JsonObject


def test_given_raw_cu_when_drafted_then_expected_canonical_fields(raw: JsonObject) -> None:
    """Unwrap values without copying CU evidence into annotation fields."""
    expected = {"segments": [{
        "category": "Events", "pages": [1, 1],
        "fields": {"items": [
            {"eventId": "A", "quantity": 12000, "unit": "kg", "date": "2026-01-01"},
            {"eventId": "B", "quantity": 8000, "unit": "kg", "date": "2026-01-02"},
        ]},
    }]}

    draft = build_ground_truth(raw)

    assert draft == expected


@pytest.mark.parametrize("layout", ["direct", "wrapped", "routed", "unrouted"])
def test_given_cu_layout_when_drafted_then_matches_adapter(layout: str, config: EvaluationConfig, raw: JsonObject) -> None:
    """Keep saved operation envelopes and segment routing consistent."""
    if layout == "wrapped":
        raw = {"status": "Succeeded", "result": raw}
    elif layout == "routed":
        raw["contents"][0]["path"] = "input1/segment1"
        raw["contents"].insert(0, {"path": "input1", "segments": [{"segmentId": "segment1"}]})
    elif layout == "unrouted":
        raw = {"contents": [{"path": "input1", "segments": [{
            "segmentId": "segment1", "category": "Other", "startPageNumber": 2, "endPageNumber": 3,
        }]}]}

    draft = build_ground_truth(raw)
    prepared = CUAdapter(config).prepare(raw, "source")

    assert draft == json.loads(prepared["response"])


def test_given_nested_values_when_drafted_then_types_and_occurrences_survive() -> None:
    """Retain nulls, empty containers, zero, false, and duplicate rows."""
    raw = {"contents": [{"fields": {
        "a/b~c": {"valueNumber": 0}, "flag": {"valueBoolean": False},
        "missing": None, "null_array": {"valueArray": None}, "null_object": {"valueObject": None},
        "empty_array": {"valueArray": []}, "empty_object": {"valueObject": {}},
        "rows": {"valueArray": [{"valueObject": {"name": {"valueString": "same"}}}] * 2},
    }}]}

    draft = build_ground_truth(raw)

    assert draft == {"segments": [{"category": None, "pages": [None, None], "fields": {
        "a/b~c": 0, "flag": False, "missing": None, "null_array": None, "null_object": None,
        "empty_array": [], "empty_object": {}, "rows": [{"name": "same"}, {"name": "same"}],
    }}]}


def test_given_mutable_value_when_draft_edited_then_raw_unchanged() -> None:
    """Detach even generic CU value containers from the caller's input."""
    raw = {"contents": [{"fields": {"details": {"value": {"names": ["original"]}}}}]}
    original = deepcopy(raw)

    draft = build_ground_truth(raw)
    draft["segments"][0]["fields"]["details"]["names"].append("corrected")

    assert raw == original


def test_given_oversized_context_when_drafted_then_no_evaluation_budget_applied(raw: JsonObject) -> None:
    """Build annotations without configuring or truncating evaluation context."""
    config = EvaluationConfig(max_context_characters=1)

    draft = build_ground_truth(raw)

    assert draft["segments"][0]["fields"]["items"][0]["quantity"] == 12000
    with pytest.raises(ValueError, match="context budget"):
        CUAdapter(config).prepare(raw, "source")


@pytest.mark.parametrize("raw", [
    {"contents": []},
    {"contents": [None]},
    {"contents": [{"fields": []}]},
    {"contents": [{"fields": {"bad": {"valueArray": "invalid"}}}]},
    {"contents": [{"fields": {"bad": {"valueUnknown": 1}}}]},
    {"contents": [{"fields": {"bad": {"valueNumber": float("nan")}}}]},
    {"contents": [{"markdown": 12}]},
    {"contents": [{"segments": [{"category": "Other"}]}]},
    {"status": "Failed", "result": {"contents": [{}]}},
])
def test_given_malformed_cu_when_drafted_then_rejected(raw: JsonObject) -> None:
    """Reject invalid CU rather than returning success-shaped annotations."""
    with pytest.raises(ValueError):
        build_ground_truth(raw)


@pytest.mark.parametrize("raw", [None, [], "not-cu"])
def test_given_unsupported_input_when_drafted_then_type_error(raw: JsonObject) -> None:
    """Require a mapping instead of accepting paths or serialized JSON."""
    with pytest.raises(TypeError, match="raw CU dictionary"):
        build_ground_truth(raw)