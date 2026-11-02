"""Verify notebook-aligned analyzer and physical-page preparation."""

import __future__

import ast
import json
import re
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
from jsonschema import Draft202012Validator, FormatChecker
from pytest_mock import MockerFixture
from typesafe_sdk import Choice

from extractly import (
    EvaluationModel,
    OnlineEvaluationConfig,
    OnlineEvaluator,
    OnlineFieldEvaluator,
    OnlineStructuralEvaluator,
    TableRule,
)
from extractly.cu._pages import prepare_pages
from extractly.eval_runner import EvaluationRunner
from extractly.evaluators._online import OnlineChecks
from extractly.types import JsonObject


@pytest.fixture
def page_input() -> tuple[JsonObject, JsonObject]:
    """Provide two synthetic physical pages with explicit record provenance."""
    schema = {"fieldSchema": {"fields": {
        "lineItems": {"type": "array", "items": {"type": "object", "properties": {"quantity": {"type": "number", "method": "extract", "description": "Use net weight."}}}},
        "deliveryMode": {"type": "string", "method": "classify", "enum": ["Truck", "Barge"], "enumDescriptions": {"Truck": "Truck evidence", "Barge": "Vessel evidence"}},
    }}}
    raw = {"status": "Succeeded", "result": {"stringEncoding": "codePoint", "contents": [{
        "kind": "document", "startPageNumber": 1, "endPageNumber": 2, "markdown": "first\nsecond",
        "pages": [{"pageNumber": 1, "spans": [{"offset": 0, "length": 5}]}, {"pageNumber": 2, "spans": [{"offset": 6, "length": 6}]}],
        "fields": {
            "lineItems": {"type": "array", "valueArray": [
                {"type": "object", "valueObject": {"quantity": {"type": "number", "valueNumber": 10, "source": "D(1,0,0,1,1)"}}},
                {"type": "object", "valueObject": {"quantity": {"type": "number", "valueNumber": 20, "source": "D(2,0,0,1,1)"}}},
            ]},
            "deliveryMode": {"type": "string", "valueString": "Truck", "source": "D(1,0,0,1,1)"},
        },
    }]}}

    return raw, schema


def test_given_two_pages_when_prepared_then_evidence_and_indices_stay_local(page_input: tuple[JsonObject, JsonObject]) -> None:
    """Keep original row identities and defer document-wide fields on each page."""
    pages = prepare_pages(*page_input, complete_document=True)

    assert [page["markdown"] for page in pages] == ["first", "second"]
    assert [page["records"][0]["path"] for page in pages] == ["lineItems[0]", "lineItems[1]"]
    assert all(not page["is_complete_document"] for page in pages)
    assert pages[0]["fields"][0]["field_schema"]["description"] == "Use net weight."
    assert pages[0]["deferred_fields"] == [{"path": "deliveryMode", "reason": "Schema requires document-level evidence."}]


def page_response(questions: dict[str, Choice]) -> SimpleNamespace:
    """Create complete deterministic Choice answers without a live client."""
    labels = {name: "present" if name.startswith("SOURCE_PRESENCE.") else "1" if name.startswith("SOURCE_COUNT.") else "pass" if name.startswith("PAGE_STRUCUTURAL_INTEGRATY.") else "tp" for name in questions}
    choices = {name: SimpleNamespace(choice=labels[name], probabilities={label: float(label == labels[name]) for label in question.criteria}, confidence=1.0) for name, question in questions.items()}
    return SimpleNamespace(answers=dict(choices), choices=choices)


def test_given_page_requests_when_executed_then_one_call_per_page(page_input: tuple[JsonObject, JsonObject], mocker: MockerFixture) -> None:
    """Do not turn two table fragments or deferred document fields into complete metrics."""
    client = mocker.Mock()
    client.system_one.side_effect = lambda *, state, questions: page_response(questions)
    checks = OnlineChecks(OnlineEvaluationConfig(), OnlineFieldEvaluator(client), OnlineStructuralEvaluator(client))
    requests = checks.prepare_pages(*page_input, complete_document=True, table_groups={"content_0_page_1": [[0]], "content_0_page_2": [[1]]})
    progress: JsonObject = {}

    result = checks.evaluate_pages(requests, progress=progress)

    assert client.system_one.call_count == 2
    assert [call.kwargs["state"]["markdown"] for call in client.system_one.call_args_list] == ["first", "second"]
    assert result["tp"] is None
    assert result["table_row_tp"] == 2
    assert result["table_count_tp"] is None
    assert progress["pages_completed"] == 2
    assert progress["document_level_evaluation_status"] == "not_evaluated"


def test_given_model_and_table_paths_when_prepared_then_config_selects_fields_and_rows(page_input: tuple[JsonObject, JsonObject], mocker: MockerFixture) -> None:
    """Apply the offline model and table selectors to page-local online requests."""
    class QuantityRow(EvaluationModel):
        """Select only the quantity within each shipment row."""

        quantity: float

    class Shipment(EvaluationModel):
        """Select the configured shipment array rather than the default lineItems."""

        shipments: list[QuantityRow]

    raw, analyzer = page_input
    definitions = analyzer["fieldSchema"]["fields"]
    definitions["shipments"] = definitions["lineItems"]
    fields = raw["result"]["contents"][0]["fields"]
    fields["shipments"] = fields["lineItems"]
    config = OnlineEvaluationConfig(extraction_model=Shipment, tables={"/segments/*/fields/shipments": TableRule()})
    checks = OnlineChecks(config, OnlineFieldEvaluator(mocker.Mock()), None)

    requests = checks.prepare_pages(raw, analyzer, complete_document=True)

    assert [set(request.field.state["fields"]) for request in requests] == [{"shipments[0].quantity"}, {"shipments[1].quantity"}]
    assert [request.context["records"][0]["path"] for request in requests] == ["shipments[0]", "shipments[1]"]
    assert all(not request.context["deferred_fields"] for request in requests)


@pytest.mark.parametrize("list_count", [1, 2])
@pytest.mark.parametrize("empty", [False, True])
def test_given_selected_lists_when_page_evaluated_then_each_list_is_one_table(list_count: int, empty: bool, page_input: tuple[JsonObject, JsonObject], mocker: MockerFixture) -> None:
    """Retain one group per selected list, including empty table containers."""
    class QuantityRow(EvaluationModel):
        """Select the numeric field in each table row."""

        quantity: float

    class SingleTable(EvaluationModel):
        """Declare one list-table in the segment fields."""

        lineItems: list[QuantityRow]

    class TwoTables(SingleTable):
        """Declare a second independent list-table."""

        other: list[QuantityRow]

    raw, analyzer = page_input
    content = raw["result"]["contents"][0]
    content["endPageNumber"] = 1
    content.pop("pages")
    items = content["fields"]["lineItems"]["valueArray"]
    if empty:
        items.clear()
    else:
        items[1]["valueObject"]["quantity"]["source"] = "D(1,0,0,1,1)"
    if list_count == 2:
        analyzer["fieldSchema"]["fields"]["other"] = analyzer["fieldSchema"]["fields"]["lineItems"]
        content["fields"]["other"] = content["fields"]["lineItems"]
    tables = {f"/segments/*/fields/{name}": TableRule() for name in ("lineItems", "other")[:list_count]}
    client = mocker.Mock()

    def answer(*, state: JsonObject, questions: dict[str, Choice]) -> SimpleNamespace:
        """Supply independent controlled source counts for the selected lists."""
        response = page_response(questions)
        for name, count in (("SOURCE_COUNT.table_row", 0 if empty else list_count * 2), ("SOURCE_COUNT.table_count", list_count)):
            if name in questions:
                label = str(count)
                response.choices[name] = SimpleNamespace(choice=label, probabilities={value: float(value == label) for value in questions[name].criteria}, confidence=1.0)
        return response

    client.system_one.side_effect = answer
    config = OnlineEvaluationConfig(extraction_model=SingleTable if list_count == 1 else TwoTables, tables=tables)
    checks = OnlineChecks(config, None, OnlineStructuralEvaluator(client))
    requests = checks.prepare_pages(raw, analyzer, complete_document=True)

    result = checks.evaluate_pages(requests, progress={})

    assert requests[0].state["extracted_table_count"] == list_count
    assert len({record["index"] for record in requests[0].context["records"]}) == (0 if empty else list_count * 2)
    assert result["table_count_tp"] == list_count
    assert result["table_row_tp"] == (0 if empty else list_count * 2)
    client.system_one.assert_called_once()


def test_given_missing_selected_list_when_page_scored_then_table_is_missing(page_input: tuple[JsonObject, JsonObject], mocker: MockerFixture) -> None:
    """Inventory source tables independently even when the candidate list is absent."""
    raw, analyzer = page_input
    content = raw["result"]["contents"][0]
    content["endPageNumber"] = 1
    content.pop("pages")
    content["fields"].pop("lineItems")
    client = mocker.Mock()
    client.system_one.side_effect = lambda *, state, questions: page_response(questions)
    checks = OnlineChecks(OnlineEvaluationConfig(tables={"/segments/*/fields/lineItems": TableRule()}), None, OnlineStructuralEvaluator(client))
    requests = checks.prepare_pages(raw, analyzer, complete_document=True)
    request = requests[0].structural
    answers = {name: {"verdict": "1" if name == "SOURCE_COUNT.table_count" else "fail", "probability": 1.0, "supported": True} for name in request.questions}

    result = checks.structural.score_page(request, answers)

    assert request.state["extracted_table_count"] == 0
    assert [result[f"table_count_{name}"] for name in ("tp", "fp", "fn")] == [0, 0, 1]
    client.system_one.assert_not_called()


def test_given_configured_cross_page_list_when_evaluated_then_document_table_count_is_deferred(page_input: tuple[JsonObject, JsonObject], mocker: MockerFixture) -> None:
    """Do not sum two page fragments of the same selected list as two tables."""
    client = mocker.Mock()
    client.system_one.side_effect = lambda *, state, questions: page_response(questions)
    checks = OnlineChecks(OnlineEvaluationConfig(tables={"/segments/*/fields/lineItems": TableRule()}), None, OnlineStructuralEvaluator(client))
    requests = checks.prepare_pages(*page_input, complete_document=True)

    result = checks.evaluate_pages(requests, progress={})

    assert [request.state["extracted_table_count"] for request in requests] == [1, 1]
    assert result["table_count_tp"] is None
    assert result["table_row_tp"] == 2


def test_given_conflicting_groups_when_lists_configured_then_rejected_before_inference(page_input: tuple[JsonObject, JsonObject], mocker: MockerFixture) -> None:
    """Prevent manual groups from silently splitting a declared list-table."""
    client = mocker.Mock()
    checks = OnlineChecks(OnlineEvaluationConfig(tables={"/segments/*/fields/lineItems": TableRule()}), None, OnlineStructuralEvaluator(client))

    with pytest.raises(ValueError, match="conflicts"):
        checks.prepare_pages(*page_input, table_groups={"content_0_page_1": []})

    client.system_one.assert_not_called()


def test_given_later_page_failure_when_executed_then_previous_page_retained(page_input: tuple[JsonObject, JsonObject], mocker: MockerFixture) -> None:
    """Propagate the original exception without discarding earlier validated judgments."""
    client = mocker.Mock()
    failure = RuntimeError("provider failure")

    def answer(*, state: JsonObject, questions: dict[str, Choice]) -> SimpleNamespace:
        """Fail only the second physical page."""
        if state["page_number"] == 2:
            raise failure
        return page_response(questions)

    client.system_one.side_effect = answer
    checks = OnlineChecks(OnlineEvaluationConfig(), OnlineFieldEvaluator(client), OnlineStructuralEvaluator(client))
    requests = checks.prepare_pages(*page_input)
    progress: JsonObject = {}

    with pytest.raises(RuntimeError, match="provider failure") as raised:
        checks.evaluate_pages(requests, progress=progress)

    assert raised.value is failure
    assert [page["evaluation_status"] for page in progress["pages"]] == ["completed", "failed"]
    assert progress["pages"][0]["fields"][0]["verdict"] == "tp"
    client.close.assert_not_called()


def test_given_analyzer_and_pages_when_library_runs_then_page_reports_are_retained(page_input: tuple[JsonObject, JsonObject], local_runner: EvaluationRunner, mocker: MockerFixture) -> None:
    """Use analyzer-defined requests without treating pages as separate source documents."""
    client = mocker.Mock()
    client.system_one.side_effect = lambda *, state, questions: page_response(questions)
    evaluator = OnlineEvaluator(OnlineEvaluationConfig(thresholds={"extraction.pooled_field_f1": 0.95}), runner=local_runner, field_evaluator=OnlineFieldEvaluator(client), structural_evaluator=OnlineStructuralEvaluator(client))
    raw, analyzer = page_input

    previews = evaluator.prepare(raw, analyzer_definition=analyzer, complete_document=True)
    client.system_one.assert_not_called()
    report = evaluator.evaluate(raw, document_id="document", analyzer_definition=analyzer, complete_document=True)

    assert [preview["state"]["markdown"] for preview in previews] == ["first", "second"]
    assert report["documents"] == 1
    assert report["metric_basis"] == "model_assisted_page_evaluation"
    assert report["coverage"]["pages_completed"] == 2
    assert report["gates"]["extraction.pooled_field_f1"] is None
    assert report["gate_status"] == "incomplete"
    assert len(evaluator.page_results["document"]["pages"]) == 2
    assert client.system_one.call_count == 2


def test_given_custom_page_settings_when_reported_then_configured_scope_and_threshold_are_retained(page_input: tuple[JsonObject, JsonObject], mocker: MockerFixture) -> None:
    """Report explicit document-wide fields and structural-only probability settings."""
    client = mocker.Mock()
    client.system_one.side_effect = lambda *, state, questions: page_response(questions)
    checks = OnlineChecks(OnlineEvaluationConfig(), None, OnlineStructuralEvaluator(client, min_probability=0.95))
    requests = checks.prepare_pages(*page_input, document_level_fields=("deliveryMode",))
    progress: JsonObject = {}

    checks.evaluate_pages(requests, progress=progress)

    assert progress["document_level_fields"] == ["deliveryMode"]
    assert all(page["review_probability_threshold"] == 0.95 for page in progress["pages"])


@pytest.fixture
def notebook_reference() -> JsonObject:
    """Load audited pure helpers and literal constants, never live notebook statements."""
    path = Path(__file__).parents[1] / "experiment" / "decision_field_schema_verification.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: JsonObject = {"json": json, "re": re, "sha256": sha256, "Choice": Choice, "Draft202012Validator": Draft202012Validator, "FormatChecker": FormatChecker, "deployment_name": "synthetic", "endpoint": "not-used"}
    constants = {"VALUE_KEYS", "DOCUMENT_LEVEL_FIELDS", "SOURCE_COUNT_LIMITS", "FIELD_VERDICTS", "EVALUATION_INSTRUCTIONS", "EXPECTED_PRESENCE_VERDICTS", "STRUCTURAL_CHECKS", "STRUCTURAL_VERDICTS", "STRUCTURAL_INSTRUCTIONS", "PAGE_STRUCUTURAL_INTEGRATY"}
    functions: list[ast.stmt] = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        for statement in ast.parse("".join(cell["source"])).body:
            if isinstance(statement, ast.FunctionDef):
                functions.append(statement)
            elif isinstance(statement, ast.Assign) and isinstance(statement.targets[0], ast.Name) and statement.targets[0].id in constants:
                namespace[statement.targets[0].id] = ast.literal_eval(statement.value)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace


@pytest.mark.parametrize("case", ["clean", "single_complete", "missing_value", "cross_page", "unresolved", "ungrouped", "duplicate_group", "missing_group", "wrong_value", "extra_value", "low_probability", "count_conflict", "skip_structure", "partial_selection", "empty_array", "schema_error"])
def test_given_notebook_scenarios_when_library_runs_then_page_and_metric_parity(case: str, page_input: tuple[JsonObject, JsonObject], notebook_reference: JsonObject, mocker: MockerFixture) -> None:
    """Compare page preparation, questions, occurrence arithmetic, and uncertainty."""
    raw, analyzer = page_input
    content = raw["result"]["contents"][0]
    records = content["fields"]["lineItems"]["valueArray"]
    quantity = records[0]["valueObject"]["quantity"]
    single = case in ("single_complete", "empty_array")
    if single:
        content["endPageNumber"] = 1
        content.pop("pages")
        records[1]["valueObject"]["quantity"]["source"] = "D(1,0,0,1,1)"
    if case == "missing_value":
        quantity.pop("valueNumber")
    elif case == "cross_page":
        quantity["source"] += "; D(2,0,0,1,1)"
    elif case == "unresolved":
        quantity.pop("source")
    elif case == "empty_array":
        records.clear()
    elif case == "schema_error":
        quantity["valueNumber"] = "not a number"
    reference = notebook_reference
    reference["ANALYZER_FIELD_SCHEMA"] = analyzer["fieldSchema"]
    reference["schema_validator"] = Draft202012Validator({"type": "object", "properties": {name: reference["to_value_schema"](node) for name, node in analyzer["fieldSchema"]["fields"].items()}, "additionalProperties": False}, format_checker=FormatChecker())
    contexts = reference["prepare_pages"](raw, complete_document=True)
    groups = {page["page_key"]: [[record["index"] for record in page["records"]]] for page in contexts} if case != "ungrouped" else {}
    if case == "duplicate_group":
        groups["content_0_page_1"] = [[0], [0]]
    elif case == "missing_group":
        groups["content_0_page_1"] = []
    selected = ["lineItems[1].quantity"] if case == "partial_selection" else None
    include_structure = case != "skip_structure"
    expected_requests = reference["build_page_requests"](contexts, selected_field_paths=selected, table_groups=groups, include_structure=include_structure)
    client = mocker.Mock()
    payloads: JsonObject = {}

    def answer(*, state: JsonObject, questions: dict[str, Choice]) -> SimpleNamespace:
        """Supply the same controlled choices to both implementations."""
        choices: JsonObject = {}
        for name, question in questions.items():
            label = "present" if name.startswith("SOURCE_PRESENCE.") else str(2 if single else 1) if name == "SOURCE_COUNT.table_row" else "1" if name.startswith("SOURCE_COUNT.") else "pass" if name.startswith("PAGE_STRUCUTURAL_INTEGRATY.") else "tn" if state["fields"][name]["value"] is None else "tp"
            if name == "lineItems[0].quantity" and case in ("wrong_value", "extra_value"):
                label = "fp"
            elif name == "SOURCE_PRESENCE.lineItems[0].quantity" and case == "extra_value":
                label = "absent"
            elif name == "SOURCE_COUNT.table_row" and case == "count_conflict":
                label = "2"
            probabilities = {item: float(item == label) for item in question.criteria}
            if name == "lineItems[0].quantity" and case == "low_probability":
                probabilities[label], probabilities["ambiguous"] = 0.55, 0.45
            choices[name] = SimpleNamespace(choice=label, probabilities=probabilities, confidence=1.0)
        payloads[f"content_{state['content_index']}_page_{state['page_number']}"] = {"answers": {name: vars(choice) for name, choice in choices.items()}}
        return SimpleNamespace(answers=dict(choices), choices=choices)

    client.system_one.side_effect = answer
    checks = OnlineChecks(OnlineEvaluationConfig(), OnlineFieldEvaluator(client), OnlineStructuralEvaluator(client) if include_structure else None)
    requests = checks.prepare_pages(raw, analyzer, complete_document=True, table_groups=groups, selected_field_paths=selected)
    for actual, expected in zip(requests, expected_requests, strict=True):
        assert actual.state == expected["state"]
        assert {name: question.model_dump(exclude_none=True) for name, question in actual.questions.items()} == {name: question.model_dump(exclude_none=True) for name, question in expected["questions"].items()}
    progress: JsonObject = {}

    metrics = checks.evaluate_pages(requests, progress=progress)

    statuses = {request.context["page_key"]: "completed" if request.questions else "not_evaluated" for request in requests}
    expected_document = reference["build_document_report"](expected_requests, payloads, statuses, threshold=0.8)
    summary = reference["build_summary_report"](expected_document, expected_requests, document_id="document", gates={})
    if case in ("empty_array", "skip_structure"):
        resolved_metrics = {f"extraction.{name}": value for name, value in {"tp": 1, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}.items()}
        expected_document["pages"][0 if case == "empty_array" else 1]["metrics"].update(resolved_metrics)
        if case == "empty_array":
            summary["document_metrics"][0].update(resolved_metrics)
    for actual, expected in zip(progress["pages"], expected_document["pages"], strict=True):
        for name in ("fields", "metrics", "deferred_fields", "deferred_records", "review_required"):
            assert actual[name] == expected[name]
        actual_structure, expected_structure = actual["PAGE_STRUCUTURAL_INTEGRATY"], expected["PAGE_STRUCUTURAL_INTEGRATY"]
        assert actual_structure["source_counts"] == expected_structure["source_counts"]
        assert {name: {key: value for key, value in check.items() if key != "reason"} for name, check in actual_structure["checks"].items()} == {name: {key: value for key, value in check.items() if key != "reason"} for name, check in expected_structure["checks"].items()}
    for name in ("tp", "fp", "fn", "precision", "recall", "f1", "exact_match", "table_row_tp", "table_row_fp", "table_row_fn", "table_count_tp", "table_count_fp", "table_count_fn"):
        assert metrics[name] == summary["document_metrics"][0][f"extraction.{name}"]