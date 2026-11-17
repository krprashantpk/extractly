"""Verify analyzer routing independently of runners and provider clients."""

from copy import deepcopy

import pytest

from extractly import select_analyzer_definitions
from extractly.types import JsonObject


@pytest.mark.parametrize("wrapped", [False, True])
def test_given_classifier_when_selected_then_each_category_has_its_own_definition(wrapped: bool, raw: JsonObject, analyzer_definitions: list[JsonObject]) -> None:
    """Identify classifiers by category configuration regardless of array order."""
    raw["contents"].append({**deepcopy(raw["contents"][0]), "path": "input2", "category": "Summary"})
    raw = {"status": "Succeeded", "result": raw} if wrapped else raw
    original = deepcopy(raw)
    definitions = [*analyzer_definitions[1:], analyzer_definitions[0]]

    selections = select_analyzer_definitions(definitions, raw)

    assert [selection.analyzer_definition for selection in selections] == analyzer_definitions[1:]
    assert all(selection.skip_reason is None for selection in selections)
    assert raw == original


@pytest.mark.parametrize("child_first", [False, True])
@pytest.mark.parametrize("null_category", [False, True])
def test_given_classifier_declaration_when_child_lacks_category_then_route_uses_declaration(child_first: bool, null_category: bool, raw: JsonObject, analyzer_definitions: list[JsonObject]) -> None:
    """Use declared categories without borrowing broader parent source text."""
    child = raw["contents"][0]
    child["path"] = "input1/segment1"
    if null_category:
        child["category"] = None
    else:
        del child["category"]
    parent = {"path": "input1", "markdown": "Must not become child evidence", "segments": [{"segmentId": "segment1", "category": "Events"}]}
    raw["contents"] = [child, parent] if child_first else [parent, child]

    selections = select_analyzer_definitions(analyzer_definitions, raw)

    assert len(selections) == 1
    assert selections[0].analyzer_definition == analyzer_definitions[1]
    assert selections[0].content.category == "Events"
    assert selections[0].content.markdown == child["markdown"]


@pytest.mark.parametrize("category", ["Unknown", "NotConfigured", None])
def test_given_unrouted_category_when_selected_then_explicit_skip(category: str | None, raw: JsonObject, analyzer_definitions: list[JsonObject]) -> None:
    """Never substitute another category's extraction rules for missing routes."""
    raw["contents"][0]["category"] = category

    selection = select_analyzer_definitions(analyzer_definitions, raw)[0]

    assert selection.analyzer_definition is None
    assert selection.skip_reason


def test_given_missing_referenced_analyzer_when_selected_then_explicit_skip(raw: JsonObject, analyzer_definitions: list[JsonObject]) -> None:
    """Retain the content outcome when its configured target was not supplied."""
    selections = select_analyzer_definitions([analyzer_definitions[0], analyzer_definitions[2]], raw)

    assert selections[0].analyzer_definition is None
    assert selections[0].skip_reason == "Classifier references an analyzer definition that was not supplied"


def test_given_declared_content_without_extraction_when_selected_then_explicit_skip(analyzer_definitions: list[JsonObject]) -> None:
    """Do not evaluate missing routed content using the classifier's source."""
    raw = {"contents": [{"path": "input1", "markdown": "Classifier-only source", "segments": [{"segmentId": "segment1", "category": "Events"}]}]}

    selection = select_analyzer_definitions(analyzer_definitions, raw)[0]

    assert selection.content.content_path == "input1/segment1"
    assert selection.content.markdown == ""
    assert selection.analyzer_definition is None
    assert selection.skip_reason == "Classified segment has no routed extraction content"


def test_given_only_classifier_when_selected_then_missing_extraction_is_skipped(raw: JsonObject, analyzer_definitions: list[JsonObject]) -> None:
    """Always inspect classifier configuration even when it is the only definition."""
    selection = select_analyzer_definitions(analyzer_definitions[:1], raw)[0]

    assert selection.analyzer_definition is None
    assert selection.skip_reason == "Classifier references an analyzer definition that was not supplied"


@pytest.mark.parametrize("identity_location", ["content", "result", "envelope"])
def test_given_no_classifier_when_selected_then_explicit_id_matches(identity_location: str, raw: JsonObject, analyzer_definitions: list[JsonObject]) -> None:
    """Accept content-level IDs and successful result or operation identities."""
    if identity_location == "content":
        raw["contents"][0]["analyzerId"] = "summary"
    elif identity_location == "result":
        raw["analyzerId"] = "summary"
    else:
        raw = {"status": "Succeeded", "analyzerId": "summary", "result": raw}

    selections = select_analyzer_definitions(analyzer_definitions[1:], raw)

    assert selections[0].analyzer_definition == analyzer_definitions[2]


def test_given_content_id_when_result_id_differs_then_content_id_takes_precedence(raw: JsonObject, analyzer_definitions: list[JsonObject]) -> None:
    """Do not overwrite a content-specific identity with the result default."""
    raw["analyzerId"] = "summary"
    raw["contents"][0]["analyzerId"] = "events"

    selection = select_analyzer_definitions(analyzer_definitions[1:], raw)[0]

    assert selection.analyzer_definition == analyzer_definitions[1]


@pytest.mark.parametrize("definitions", [[], [{"fields": {"quantity": {"type": "number"}}}]])
def test_given_zero_or_one_definition_when_selected_then_no_identity_is_required(definitions: list[JsonObject], raw: JsonObject) -> None:
    """Keep custom evaluator and standalone field-schema usage possible."""
    selection = select_analyzer_definitions(definitions, raw)[0]

    assert selection.analyzer_definition == (definitions[0] if definitions else None)
    assert selection.skip_reason is None


@pytest.mark.parametrize("analyzer_id", [None, "missing", "", 4])
def test_given_multiple_unclassified_definitions_when_identity_unusable_then_rejected(analyzer_id: object, raw: JsonObject, analyzer_definitions: list[JsonObject]) -> None:
    """Reject ambiguous selection instead of choosing the first definition."""
    raw["contents"][0]["analyzerId"] = analyzer_id

    with pytest.raises(ValueError, match="matching content or result analyzerId"):
        select_analyzer_definitions(analyzer_definitions[1:], raw)


@pytest.mark.parametrize("definitions", [
    [{"analyzerId": "same"}, {"analyzerId": "same"}],
    [{"analyzerId": "one"}, {}],
    [{"analyzerId": "one", "config": {"contentCategories": {"A": {}}}}, {"analyzerId": "two", "config": {"contentCategories": {"B": {}}}}],
    [{"config": {"contentCategories": []}}],
    [{"config": {"contentCategories": {"A": {"analyzerId": ""}}}}],
])
def test_given_invalid_definitions_when_selected_then_rejected(definitions: list[JsonObject], raw: JsonObject) -> None:
    """Reject malformed or ambiguous configuration before evaluation."""
    with pytest.raises(ValueError):
        select_analyzer_definitions(definitions, raw)
