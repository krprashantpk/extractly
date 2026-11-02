"""Test Decision judgments and reference-free field and structural evaluation."""

import json
from types import SimpleNamespace

import pytest
from pytest_mock import MockerFixture
from typesafe_sdk import Choice

from extractly import CUAdapter, EvaluationConfig, EvaluationModel, OnlineEvaluationConfig, OnlineEvaluator, TableRule
from extractly.eval_runner import EvaluationRunner
from extractly.evaluators._decision import judge
from extractly.evaluators._online import OnlineChecks
from extractly.evaluators.online_field import OnlineFieldEvaluator
from extractly.evaluators.online_structural import OnlineStructuralEvaluator
from extractly.types import JsonObject


@pytest.mark.parametrize("probability", [True, -0.1, 1.1, float("nan"), float("inf")])
def test_given_invalid_probabilities_when_judged_then_rejected(probability: object, mocker: MockerFixture) -> None:
    """Reject nonnumeric and nonfinite service probabilities instead of approving them."""
    client = mocker.Mock()
    client.system_one.return_value = SimpleNamespace(answers={"field": {}}, choices={"field": SimpleNamespace(choice="tp", probabilities={"tp": probability}, confidence=1.0)})

    with pytest.raises(RuntimeError, match="invalid probabilities"):
        judge(client, {}, {"field": Choice(criteria={"tp": "Supported value"})}, min_probability=0.8)


def test_given_missing_answers_when_judged_then_rejected(mocker: MockerFixture) -> None:
    """Do not turn refused or incomplete provider responses into valid judgments."""
    client = mocker.Mock()
    client.system_one.return_value = SimpleNamespace(answers={}, choices={})

    with pytest.raises(RuntimeError, match="exactly one"):
        judge(client, {}, {"field": Choice(criteria={"tp": "Supported value"})}, min_probability=0.8)


def test_given_no_questions_when_judged_then_client_is_not_used(mocker: MockerFixture) -> None:
    """Skip empty requests without constructing or closing provider resources."""
    client = mocker.Mock()

    assert judge(client, {}, {}, min_probability=0.8) == {}
    client.system_one.assert_not_called()
    client.close.assert_not_called()


def decision_response(questions: dict[str, Choice], labels: dict[str, str]) -> SimpleNamespace:
    """Create a deterministic provider response with complete Choice distributions."""
    choices = {name: SimpleNamespace(choice=labels[name], probabilities={label: float(label == labels[name]) for label in question.criteria}, confidence=1.0) for name, question in questions.items()}
    return SimpleNamespace(answers=dict(choices), choices=choices)


@pytest.fixture
def analyzer_definition(raw: JsonObject) -> JsonObject:
    """Supply the original analyzer types instead of inferring policy from predictions."""
    properties = {name: {"type": kind} for name, kind in (("eventId", "string"), ("quantity", "number"), ("unit", "string"), ("date", "date"))}
    for row in raw["contents"][0]["fields"]["items"]["valueArray"]:
        row["type"] = "object"
        for name, field in row["valueObject"].items():
            field["type"] = properties[name]["type"]
    return {"fieldSchema": {"fields": {"items": {"type": "array", "items": {"type": "object", "properties": properties}}}}}


def successful_page_response(questions: dict[str, Choice]) -> SimpleNamespace:
    """Return coherent field, structure, and source-count answers for two fixture records."""
    labels = {name: "present" if name.startswith("SOURCE_PRESENCE.") else "2" if name == "SOURCE_COUNT.table_row" else "1" if name.startswith("SOURCE_COUNT.") else "pass" if name.startswith("PAGE_STRUCUTURAL_INTEGRATY.") else "tp" for name in questions}
    return decision_response(questions, labels)


@pytest.mark.parametrize(("verdict", "presence", "counts"), [("tp", "present", (1, 0, 0)), ("fp", "present", (0, 1, 1)), ("fp", "absent", (0, 1, 0)), ("ambiguous", "present", (None, None, None))])
def test_given_field_judgments_when_scored_then_directional_counts(verdict: str, presence: str, counts: tuple[int | None, ...], mocker: MockerFixture) -> None:
    """Distinguish wrong values, extra values, and unresolved evidence."""
    client = mocker.Mock()
    client.system_one.side_effect = lambda *, state, questions: decision_response(questions, {name: verdict if name.startswith("field:") else presence for name in questions})
    evaluator = OnlineFieldEvaluator(client)

    result = evaluator([{"field_path": "/quantity", "response": '{"value": 0}'}], "Quantity: 1 kg", "{}")

    assert tuple(result[name] for name in ("tp", "fp", "fn")) == counts
    client.close.assert_not_called()


@pytest.mark.parametrize(("verdict", "counts"), [("fn", (0, 0, 1)), ("tn", (0, 0, 0))])
def test_given_absent_field_when_judged_then_absence_is_not_always_missing(verdict: str, counts: tuple[int, ...], mocker: MockerFixture) -> None:
    """Preserve true negatives without adding them to positive occurrence totals."""
    client = mocker.Mock()
    client.system_one.side_effect = lambda *, state, questions: decision_response(questions, dict.fromkeys(questions, verdict))

    result = OnlineFieldEvaluator(client)([{"field_path": "/quantity", "response": '{"presence": "absent"}'}], "No quantity recorded", "{}")

    assert tuple(result[name] for name in ("tp", "fp", "fn")) == counts


def test_given_adapter_slots_when_judged_then_paths_and_row_context_preserved(raw: JsonObject, mocker: MockerFixture) -> None:
    """Reuse the library's adapter instead of introducing notebook-specific field parsing."""
    adapter = CUAdapter(EvaluationConfig())
    record = adapter.prepare(raw, "document")
    rows = adapter.field_rows(record)

    request = OnlineFieldEvaluator(mocker.Mock()).prepare(rows, record["context"], "{}")

    assert set(request.state["fields"]) == {row["field_path"] for row in rows}
    assert all(field["row_context"] != "{}" for field in request.state["fields"].values())


@pytest.mark.parametrize(("source_rows", "row_verdict", "counts"), [("2", "pass", (2, 0, 0)), ("3", "fail", (2, 0, 1)), ("1", "fail", (1, 1, 0)), ("overflow", "ambiguous", (None, None, None))])
def test_given_source_counts_when_structure_scored_then_positional_occurrences(source_rows: str, row_verdict: str, counts: tuple[int | None, ...], raw: JsonObject, config: EvaluationConfig, mocker: MockerFixture) -> None:
    """Measure presence separately from field values and preserve unknown counts."""
    record = CUAdapter(config).prepare(raw, "document")
    client = mocker.Mock()
    client.system_one.side_effect = lambda *, state, questions: decision_response(questions, {name: source_rows if name.startswith("count:table_row:") else "1" if name.startswith("count:") else row_verdict if name == "structure:row_count" else "pass" for name in questions})

    result = OnlineStructuralEvaluator(client)(record["response"], record["context"], config.model_dump_json(include={"extraction_model", "tables"}))

    assert tuple(result[f"table_row_{name}"] for name in ("tp", "fp", "fn")) == counts
    assert result["table_count_tp"] == 1
    client.close.assert_not_called()


def test_given_flat_rows_without_table_rules_when_judged_then_tables_unassessable(raw: JsonObject, mocker: MockerFixture) -> None:
    """Never infer extracted table groups from source markup or a flat row list."""
    record = CUAdapter(EvaluationConfig()).prepare(raw, "document")
    client = mocker.Mock()

    result = OnlineStructuralEvaluator(client)(record["response"], record["context"], "{}")

    assert result["table_count_tp"] is None
    assert result["checks"]["table_identity"]["verdict"] == "not_assessable"
    assert result["review_required"] is True
    client.system_one.assert_not_called()


@pytest.mark.parametrize("list_count", [1, 2])
def test_given_configured_row_lists_when_structure_scored_then_containers_are_tables(list_count: int, mocker: MockerFixture) -> None:
    """Use each selected list as a table without a separate parent-array rule."""
    fields = {name: [{"quantity": 1}, {"quantity": 2}] for name in ("items", "other")[:list_count]}
    config = EvaluationConfig(tables={f"/segments/*/fields/{name}": TableRule() for name in fields})
    client = mocker.Mock()
    client.system_one.side_effect = lambda *, state, questions: decision_response(questions, {name: "2" if name.startswith("count:table_row:") else "1" if name.startswith("count:") else "pass" for name in questions})

    result = OnlineStructuralEvaluator(client)(json.dumps({"segments": [{"fields": fields}]}), "Independent source tables and rows", config.model_dump_json(include={"tables"}))

    assert result["table_count_tp"] == list_count
    assert result["table_row_tp"] == list_count * 2
    assert result["checks"]["row_table_association"]["verdict"] == "pass"
    assert result["table_count_f1"] == 1.0
    client.close.assert_not_called()


def test_given_duplicate_rows_when_counts_match_then_integrity_still_fails(raw: JsonObject, config: EvaluationConfig, mocker: MockerFixture) -> None:
    """Equal presence counts never override a separate duplicate occurrence failure."""
    record = CUAdapter(config).prepare(raw, "document")
    client = mocker.Mock()
    client.system_one.side_effect = lambda *, state, questions: decision_response(questions, {name: "2" if name.startswith("count:table_row:") else "1" if name.startswith("count:") else "fail" if name == "structure:duplicate_rows" else "pass" for name in questions})

    result = OnlineStructuralEvaluator(client)(record["response"], record["context"], config.model_dump_json(include={"extraction_model", "tables"}))

    assert result["table_row_f1"] == 1.0
    assert result["review_required"] is True
    assert result["rows_complete"] is False


def test_given_both_checks_when_combined_then_one_call_and_separate_results(raw: JsonObject, config: EvaluationConfig, mocker: MockerFixture) -> None:
    """Combine field and structure questions without duplicating client requests or parsing."""
    adapter = CUAdapter(config)
    record = adapter.prepare(raw, "document")
    client = mocker.Mock()
    client.system_one.side_effect = lambda *, state, questions: decision_response(questions, {name: "tp" if name.startswith("field:") else "present" if name.startswith("presence:") else "2" if name.startswith("count:table_row:") else "1" if name.startswith("count:") else "pass" for name in questions})
    evaluator = OnlineChecks(config, OnlineFieldEvaluator(client), OnlineStructuralEvaluator(client))

    result = evaluator(record["response"], record["context"], config.model_dump_json(include={"extraction_model", "tables"}), adapter.field_rows(record))

    assert result["tp"] == 8
    assert result["table_row_tp"] == 2
    assert result["table_count_tp"] == 1
    assert result["review_required"] is False
    client.system_one.assert_called_once()
    client.close.assert_not_called()


def test_given_different_clients_when_combined_then_rejected(config: EvaluationConfig, mocker: MockerFixture) -> None:
    """Keep combined request ownership explicit instead of silently choosing a client."""
    with pytest.raises(ValueError, match="share the same"):
        OnlineChecks(config, OnlineFieldEvaluator(mocker.Mock()), OnlineStructuralEvaluator(mocker.Mock()))


def test_given_decision_checks_when_online_runs_then_offline_shaped_report(raw: JsonObject, config: EvaluationConfig, analyzer_definition: JsonObject, local_runner: EvaluationRunner, mocker: MockerFixture) -> None:
    """Reuse schema, runner, metric gates, and document reporting without reference labels."""
    client = mocker.Mock()
    client.system_one.side_effect = lambda *, state, questions: successful_page_response(questions)
    online = OnlineEvaluationConfig.model_validate(config.model_dump())
    evaluator = OnlineEvaluator(online, runner=local_runner, field_evaluator=OnlineFieldEvaluator(client), structural_evaluator=OnlineStructuralEvaluator(client))

    report = evaluator.evaluate(raw, document_id="document", analyzer_definition=analyzer_definition, complete_document=True, table_groups={"content_0_page_1": [[0, 1]]})

    assert report["mode"] == "online"
    assert report["metrics"]["extraction.pooled_field_f1"] == 1.0
    assert report["metrics"]["extraction.table_row_tp"] == 2
    assert report["metrics"]["extraction.table_count_tp"] == 1
    assert report["document_metrics"][0]["id"] == "document"
    assert report["metric_basis"] == "model_assisted_page_evaluation"
    assert report["document_metrics"][0]["review_required"] is False
    assert "review_recommendation" not in report
    client.system_one.assert_called_once()
    client.close.assert_not_called()


@pytest.mark.parametrize(("uncertain_fields", "verdict", "row_check"), [(1, "ambiguous", None), (1, "fn", None), (8, "ambiguous", None), (1, "fn", "duplicate_rows"), (0, "tp", "missing_rows"), (0, "tp", "extra_rows")])
def test_given_uncertain_document_when_batch_runs_then_resolved_counts_and_review_retained(
    uncertain_fields: int,
    verdict: str,
    row_check: str | None,
    raw: JsonObject,
    config: EvaluationConfig,
    analyzer_definition: JsonObject,
    local_runner: EvaluationRunner,
    mocker: MockerFixture,
) -> None:
    """Exclude uncertain fields from counts without losing document review or resolved judgments."""
    client = mocker.Mock()
    states: list[JsonObject] = []

    def answer(*, state: JsonObject, questions: dict[str, Choice]) -> SimpleNamespace:
        """Leave selected second-document fields ambiguous or below the probability threshold."""
        states.append(state)
        uncertain = {name for name in questions if name.startswith("items[")} if len(states) == 2 else set()
        uncertain = set(sorted(uncertain)[:uncertain_fields])
        labels = {name: verdict if name in uncertain else "tp" if name.startswith("items[") else "present" if name.startswith("SOURCE_PRESENCE.") else "2" if name == "SOURCE_COUNT.table_row" else "1" if name.startswith("SOURCE_COUNT.") else "pass" for name in questions}
        response = decision_response(questions, labels)
        if verdict == "fn":
            for name in uncertain:
                response.choices[name].probabilities = {label: 0.7 if label == "fn" else 0.3 if label == "tp" else 0.0 for label in questions[name].criteria}
        if row_check is not None and len(states) == 2:
            name = f"PAGE_STRUCUTURAL_INTEGRATY.{row_check}"
            response.choices[name].choice = "not_assessable"
            response.choices[name].probabilities = {label: 0.5 if label in ("not_assessable", "pass") else 0.0 for label in questions[name].criteria}
        return response

    client.system_one.side_effect = answer
    evaluator = OnlineEvaluator(OnlineEvaluationConfig.model_validate(config.model_dump()), runner=local_runner, field_evaluator=OnlineFieldEvaluator(client), structural_evaluator=OnlineStructuralEvaluator(client))

    report = evaluator.run([{"id": identity, "cu_result": raw, "analyzer_definition": analyzer_definition, "complete_document": True, "ground_truth": {"PRIVATE_REFERENCE": True}} for identity in ("first", "second")])

    assert report["documents"] == 2
    assert report["metrics"]["extraction.tp"] == (16 - uncertain_fields) / 2
    assert report["metrics"]["extraction.fp"] == report["metrics"]["extraction.fn"] == 0
    assert report["metrics"]["extraction.pooled_field_f1"] == 1.0
    assert report["document_metrics"][0]["extraction.tp"] == 8
    assert report["document_metrics"][1]["extraction.tp"] == 8 - uncertain_fields
    assert report["document_metrics"][1]["extraction.fp"] == report["document_metrics"][1]["extraction.fn"] == 0
    page = evaluator.page_results["second"]["pages"][0]
    assert page["field_metrics_complete"] is False
    assert sum(field["occurrence_counts"] is None for field in page["fields"]) == uncertain_fields
    assert page["assessed_field_counts"] == {"tp": 8 - uncertain_fields, "fp": 0, "fn": 0}
    assert [document["review_required"] for document in report["document_metrics"]] == [document["extraction.review_required"] or not document["schema.valid"] for document in report["document_metrics"]]
    assert report["document_metrics"][1]["review_required"] is True
    assert "review_recommendation" not in report
    assert "PRIVATE_REFERENCE" not in str(states)


def test_given_optional_absent_slot_when_decision_prepared_then_included(raw: JsonObject, mocker: MockerFixture) -> None:
    """Reuse optional slot enumeration without changing legacy adapter selection."""
    class Fields(EvaluationModel):
        """Select an optional absent field for source-presence evaluation."""

        note: str | None = None

    adapter = CUAdapter(EvaluationConfig(extraction_model=Fields))
    record = adapter.prepare(raw, "document")

    assert adapter.field_rows(record) == []
    rows = adapter.field_rows(record, include_optional_absences=True)
    request = OnlineFieldEvaluator(mocker.Mock()).prepare(rows, record["context"], "{}")

    assert set(request.state["fields"]) == {"/segments/0/fields/note"}
    assert request.state["fields"]["/segments/0/fields/note"]["presence"] == "absent"


def test_given_multiple_pages_without_spans_when_library_runs_then_rejected(raw: JsonObject, config: EvaluationConfig, analyzer_definition: JsonObject, local_runner: EvaluationRunner, mocker: MockerFixture) -> None:
    """Reject unsafe page slicing rather than silently judging a whole document."""
    raw["contents"][0]["endPageNumber"] = 2
    raw["contents"][0]["markdown"] += "\nPage two: source document summary"
    client = mocker.Mock()
    client.system_one.side_effect = lambda *, state, questions: decision_response(questions, {name: "tp" if name.startswith("field:") else "present" if name.startswith("presence:") else "2" if name.startswith("count:table_row:") else "1" if name.startswith("count:") else "pass" for name in questions})
    evaluator = OnlineEvaluator(OnlineEvaluationConfig.model_validate(config.model_dump()), runner=local_runner, field_evaluator=OnlineFieldEvaluator(client), structural_evaluator=OnlineStructuralEvaluator(client))

    with pytest.raises(ValueError, match="spans"):
        evaluator.evaluate(raw, document_id="two-pages", analyzer_definition=analyzer_definition)

    client.system_one.assert_not_called()


def test_given_later_invalid_document_when_prepared_then_no_decision_calls(raw: JsonObject, analyzer_definition: JsonObject, local_runner: EvaluationRunner, mocker: MockerFixture) -> None:
    """Complete batch validation before any external Decision operation."""
    client = mocker.Mock()
    evaluator = OnlineEvaluator(OnlineEvaluationConfig(), runner=local_runner, field_evaluator=OnlineFieldEvaluator(client))

    with pytest.raises(ValueError, match="contents"):
        evaluator.run([{"id": "valid", "cu_result": raw, "analyzer_definition": analyzer_definition}, {"id": "invalid", "cu_result": {}}])

    client.system_one.assert_not_called()


def test_given_provider_failure_when_online_runs_then_error_propagates(raw: JsonObject, analyzer_definition: JsonObject, local_runner: EvaluationRunner, mocker: MockerFixture) -> None:
    """Preserve caller retry policy and never return a success-shaped report on failure."""
    client = mocker.Mock()
    failure = RuntimeError("provider failure")
    client.system_one.side_effect = failure
    evaluator = OnlineEvaluator(OnlineEvaluationConfig(), runner=local_runner, field_evaluator=OnlineFieldEvaluator(client))

    with pytest.raises(RuntimeError, match="provider failure") as raised:
        evaluator.evaluate(raw, document_id="document", analyzer_definition=analyzer_definition)

    assert raised.value is failure
    client.close.assert_not_called()


@pytest.mark.parametrize("threshold", [True, -0.1, 1.1, float("nan")])
def test_given_invalid_threshold_when_constructed_then_rejected(threshold: float, mocker: MockerFixture) -> None:
    """Validate probability policy before an external call can occur."""
    with pytest.raises((TypeError, ValueError), match="min_probability"):
        OnlineFieldEvaluator(mocker.Mock(), min_probability=threshold)


@pytest.mark.parametrize("row", [{"field_path": "/value"}, {"field_path": "/value", "response": 1}])
def test_given_malformed_field_row_when_prepared_then_rejected(row: JsonObject, mocker: MockerFixture) -> None:
    """Reject malformed slot inputs before using caller-owned service resources."""
    client = mocker.Mock()

    with pytest.raises(TypeError, match="field_path and response"):
        OnlineFieldEvaluator(client)([row], "source", "{}")

    client.system_one.assert_not_called()