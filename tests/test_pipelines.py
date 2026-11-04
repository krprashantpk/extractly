"""Verify injectable metric-based pipelines without service calls."""

import json
from copy import deepcopy
from pathlib import Path

import pytest
from openai import OpenAI
from pytest_mock import MockerFixture

import extractly._pipeline as pipeline_module
from extractly import EvaluationModel, LLMFieldEvaluator, LLMStructuralEvaluator, SchemaEvaluator
from extractly.eval_runner import LocalRunner, RunnerRequest, RunnerResult
from extractly.offline_pipeline import OfflineEvaluationConfig, OfflineEvaluationPipeline
from extractly.online_pipeline import OnlineEvaluationConfig, OnlineEvaluationPipeline
from extractly.types import JsonObject


@pytest.mark.parametrize("mode", ["offline", "online"])
def test_given_pipeline_mode_when_dispatched_then_matching_workflow_runs(mode: str, mocker: MockerFixture) -> None:
    """Keep the shared entry point limited to selecting the requested workflow."""
    offline = mocker.patch.object(pipeline_module, "run_offline_pipeline", return_value={"mode": "offline"})
    online = mocker.patch.object(pipeline_module, "run_online_pipeline", return_value={"mode": "online"})
    config = OfflineEvaluationConfig() if mode == "offline" else OnlineEvaluationConfig()
    documents: list[JsonObject] | JsonObject = [] if mode == "offline" else {"id": "d"}
    evaluators = {"schema": SchemaEvaluator()}
    runner = LocalRunner()

    report = pipeline_module.run_pipeline(config, documents, evaluators, runner, mode=mode)

    selected, unused = (offline, online) if mode == "offline" else (online, offline)
    selected.assert_called_once_with(config, documents, evaluators, runner)
    unused.assert_not_called()
    assert report == {"mode": mode}


def test_given_unknown_mode_when_dispatched_then_rejected() -> None:
    """Do not silently treat an unsupported mode as online evaluation."""
    with pytest.raises(ValueError, match="mode must be offline or online"):
        pipeline_module.run_pipeline(OnlineEvaluationConfig(), [], {"schema": SchemaEvaluator()}, LocalRunner(), mode="unknown")


@pytest.mark.parametrize("ground_truth", [None, [], "invalid"])
def test_given_offline_document_without_valid_gold_when_run_then_runner_not_called(ground_truth: object, raw: JsonObject, mocker: MockerFixture) -> None:
    """Require independent dictionary references in the offline workflow."""
    runner = LocalRunner()
    run = mocker.patch.object(runner, "run")

    with pytest.raises(ValueError, match="requires ground_truth"):
        pipeline_module.run_offline_pipeline(OfflineEvaluationConfig(), [{"id": "d", "cu_result": raw, "ground_truth": ground_truth}], {"schema": SchemaEvaluator()}, runner)

    run.assert_not_called()


def test_given_valid_offline_document_when_run_then_parsed_fields_and_gold_are_forwarded(raw: JsonObject) -> None:
    """Keep independent references and source evidence in the offline workflow."""
    received: list[JsonObject] = []

    def custom(extracted_values: JsonObject, ground_truth: JsonObject, markdown: str, **kwargs: object) -> JsonObject:
        """Capture the supported offline inputs without a provider request."""
        assert "analyzer_definition" not in kwargs
        received.append({"quantity": extracted_values["items"][0]["quantity"], "ground_truth": ground_truth, "markdown": markdown})
        return {"metrics": {"score": 1}}

    gold = {"items": [{"quantity": 42}]}
    document = {"id": "d", "cu_result": raw, "ground_truth": gold, "analyzer_definition": {"description": "Ignored offline"}}
    original = deepcopy(document)

    report = pipeline_module.run_offline_pipeline(OfflineEvaluationConfig(), [document], {"custom": custom}, LocalRunner())

    assert received == [{"quantity": 12000, "ground_truth": gold, "markdown": raw["contents"][0]["markdown"]}]
    assert report["metrics"] == {"custom": {"score": 1.0}}
    assert document == original


@pytest.mark.parametrize("mode", ["offline", "online"])
def test_given_excess_context_when_run_then_neither_workflow_truncates(mode: str, raw: JsonObject, mocker: MockerFixture) -> None:
    """Preserve context-budget enforcement after extracting the mode paths."""
    config = OfflineEvaluationConfig(max_context_characters=1) if mode == "offline" else OnlineEvaluationConfig(max_context_characters=1)
    runner = LocalRunner()
    run = mocker.patch.object(runner, "run")

    document = {"id": "d", "cu_result": raw, "ground_truth": {}}
    documents = [document] if mode == "offline" else document
    with pytest.raises(ValueError, match="never truncate"):
        pipeline_module.run_pipeline(config, documents, {"schema": SchemaEvaluator()}, runner, mode=mode)

    run.assert_not_called()


def test_given_default_offline_checks_when_run_then_reports_are_namespaced(raw: JsonObject, quantity_schema: type[EvaluationModel]) -> None:
    """Keep default metric namespaces and pooled field scores separate."""
    pipeline = OfflineEvaluationPipeline(OfflineEvaluationConfig(extraction_model=quantity_schema), runner=LocalRunner())
    gold = {"items": [{"quantity": 12000}, {"quantity": 8000}]}

    report = pipeline.run(raw, document_id="d", ground_truth=gold)

    assert set(report["metrics"]) == {"field", "structural", "schema"}
    assert report["metrics"]["field"]["extraction.pooled_field_f1"] == 1.0
    assert report["metrics"]["structural"]["extraction.table_row.tp"] == 2
    assert report["document_metrics"][0]["metrics"]["schema"]["valid"] == 1


def test_given_custom_evaluator_when_run_then_only_explicit_metrics_are_aggregated(raw: JsonObject) -> None:
    """Do not turn arbitrary diagnostic numbers into aggregate scores."""
    def custom(extracted_values: JsonObject) -> JsonObject:
        """Return one metric and a numeric diagnostic."""
        return {"metrics": {"score": 187 / 188, "undefined": None}, "diagnostic": 99}

    pipeline = OfflineEvaluationPipeline(OfflineEvaluationConfig(thresholds={"application.score": 0.995}), runner=LocalRunner(), evaluators={"application": custom})

    report = pipeline.run(raw, document_id="d", ground_truth={})

    assert report["metrics"] == {"application": {"score": 0.995, "undefined": None}}
    assert report["gates"]["application.score"] is False
    assert "diagnostic" not in json.dumps(report)


def test_given_online_record_with_gold_when_run_then_evaluator_never_receives_gold(raw: JsonObject) -> None:
    """Keep online evaluation reference-free even for arbitrary custom callables."""
    received: list[JsonObject] = []

    def custom(**kwargs: object) -> JsonObject:
        """Capture injected call-time inputs without inference."""
        received.append(kwargs)
        return {"metrics": {"score": 1}}

    pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(), runner=LocalRunner(), evaluators={"custom": custom})

    analyzer = {"description": "Online analyzer"}
    report = pipeline_module.run_online_pipeline(pipeline.config, {"id": "d", "cu_result": raw, "ground_truth": {"secret": "gold"}, "analyzer_definition": analyzer}, pipeline.evaluators, pipeline.runner)

    assert report["metrics"] == {"custom": {"score": 1.0}}
    assert received and "ground_truth" not in received[0]
    assert received[0]["analyzer_definition"] == analyzer


def test_given_old_style_output_when_run_then_metrics_contract_is_required(raw: JsonObject) -> None:
    """Reject success-shaped diagnostics that lack an explicit metrics dictionary."""
    def old_style(**kwargs: object) -> JsonObject:
        """Demonstrate an unsupported pre-migration evaluator result."""
        return {"score": 1}

    pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(), runner=LocalRunner(), evaluators={"custom": old_style})

    with pytest.raises(RuntimeError, match="metrics dictionary"):
        pipeline.evaluate(raw, document_id="d")


def test_given_different_document_sizes_when_aggregated_then_mean_and_pooled_scores_differ(raw: JsonObject) -> None:
    """Preserve occurrence pooling rather than averaging rounded document F1 values."""
    results = iter([{"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5, "f1": 0.5}, {"tp": 9, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0}])

    def custom(**kwargs: object) -> JsonObject:
        """Return independently controlled occurrence counts for each document."""
        return {"metrics": {f"extraction.document.{key}": value for key, value in next(results).items()}}

    pipeline = OfflineEvaluationPipeline(OfflineEvaluationConfig(), runner=LocalRunner(), evaluators={"field": custom})

    report = pipeline._evaluate([{"id": identity, "cu_result": raw, "ground_truth": {}} for identity in ("a", "b")])

    assert report["metrics"]["field"]["extraction.mean_document_f1"] == 0.75
    assert report["metrics"]["field"]["extraction.pooled_field_f1"] == 0.909
    assert report["metrics"]["field"]["extraction.document.tp"] == 5
    assert [document["review_required"] for document in report["document_metrics"]] == [True, False]


def test_given_client_and_deployment_when_online_defaults_created_then_client_is_retained(mocker: MockerFixture) -> None:
    """Construct the default evaluator list without creating or closing a client."""
    client = mocker.Mock(spec=OpenAI)

    pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(), runner=LocalRunner(), client=client, deployment="test-model")

    assert isinstance(pipeline.evaluators["field"], LLMFieldEvaluator)
    assert isinstance(pipeline.evaluators["structural"], LLMStructuralEvaluator)
    assert isinstance(pipeline.evaluators["schema"], SchemaEvaluator)
    assert pipeline.evaluators["field"].client is client
    client.close.assert_not_called()


def test_given_online_defaults_when_run_then_reference_free_evaluator_inputs_are_bound(raw: JsonObject, online_config: OnlineEvaluationConfig, mocker: MockerFixture) -> None:
    """Exercise all default evaluators without making a provider request."""
    client = mocker.Mock(spec=OpenAI)
    field = mocker.patch.object(LLMFieldEvaluator, "__call__", return_value={"metrics": {"extraction.document.tp": 2, "extraction.document.fp": 0, "extraction.document.fn": 0, "extraction.document.f1": 1.0}})
    structural = mocker.patch.object(LLMStructuralEvaluator, "__call__", return_value={"metrics": {"extraction.table_row.tp": 2, "extraction.table_row.fp": 0, "extraction.table_row.fn": 0}, "review_required": True})
    pipeline = OnlineEvaluationPipeline(online_config, runner=LocalRunner(), client=client, deployment="test-model")
    analyzer = {"fieldSchema": {"fields": {"items": {"type": "array"}}}}

    report = pipeline.evaluate(raw, document_id="d", analyzer_definition=analyzer)

    assert set(report["metrics"]) == {"field", "structural", "schema"}
    assert report["review_required"] is True
    assert field.call_args.kwargs["analyzer_definition"] == analyzer
    assert "ground_truth" not in structural.call_args.kwargs
    assert field.call_args.kwargs["extracted_values"]["items"][0]["quantity"] == 12000
    assert field.call_args.kwargs["markdown"] == raw["contents"][0]["markdown"]


@pytest.mark.parametrize("invalid", [{}, {"schema": None}, {"": lambda: {}}, {"bad.name": lambda: {}}])
def test_given_invalid_evaluator_mapping_when_constructed_then_rejected(invalid: JsonObject) -> None:
    """Reject empty mappings, ambiguous names, and noncallable dependencies early."""
    with pytest.raises((TypeError, ValueError)):
        OfflineEvaluationPipeline(OfflineEvaluationConfig(), runner=LocalRunner(), evaluators=invalid)


@pytest.mark.parametrize("metric", ["invalid", {}, float("inf"), float("nan")])
def test_given_invalid_metric_when_evaluated_then_contract_failure_is_visible(metric: object, raw: JsonObject) -> None:
    """Never treat nonnumeric or nonfinite metrics as an evaluation success."""
    def invalid(**kwargs: object) -> JsonObject:
        """Return a malformed metric for boundary validation."""
        return {"metrics": {"score": metric}}

    pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(), runner=LocalRunner(), evaluators={"custom": invalid})

    with pytest.raises(RuntimeError, match="finite"):
        pipeline.evaluate(raw, document_id="d")


@pytest.mark.parametrize("document", [{"id": "d"}, {"id": ""}, {"id": "d", "cu_result": {}}])
def test_given_invalid_documents_when_run_then_rejected(document: JsonObject) -> None:
    """Validate source envelopes before any evaluator request is made."""
    pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(), runner=LocalRunner(), evaluators={"custom": lambda **kwargs: {"metrics": {"score": 1}}})

    with pytest.raises(ValueError):
        pipeline_module.run_online_pipeline(pipeline.config, document, pipeline.evaluators, pipeline.runner)


def test_given_online_batch_when_run_then_rejected_before_runner(raw: JsonObject, mocker: MockerFixture) -> None:
    """Reject multiple files instead of silently evaluating only the first."""
    runner = LocalRunner()
    run = mocker.patch.object(runner, "run")
    pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(), runner=runner, evaluators={"custom": lambda **kwargs: {"metrics": {"score": 1}}})
    document = {"id": "d", "cu_result": raw}

    with pytest.raises(TypeError, match="dictionary"):
        pipeline.run([document, document], document_id="d")  # type: ignore[arg-type]

    run.assert_not_called()


def test_given_provider_failure_when_run_then_original_error_and_cleanup_are_preserved(raw: JsonObject, mocker: MockerFixture) -> None:
    """Clean temporary evidence files while preserving caller exception identity."""
    paths: list[Path] = []
    failure = ConnectionError("Synthetic runner failure")
    runner = LocalRunner()

    def fail(request: RunnerRequest) -> RunnerResult:
        """Capture the temporary dataset and propagate a provider failure."""
        paths.append(request.data)
        raise failure

    mocker.patch.object(runner, "run", side_effect=fail)
    pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(), runner=runner, evaluators={"custom": lambda **kwargs: {"metrics": {"score": 1}}})

    with pytest.raises(ConnectionError) as caught:
        pipeline.evaluate(raw, document_id="d")

    assert caught.value is failure
    assert paths and not paths[0].exists()


def test_given_unknown_occurrence_count_when_aggregated_then_pooled_scores_remain_unknown(raw: JsonObject) -> None:
    """Do not manufacture perfect pooled scores for incomplete evidence."""
    def unknown(**kwargs: object) -> JsonObject:
        """Report independently unresolved occurrence counts."""
        return {"metrics": {"extraction.document.tp": None, "extraction.document.fp": None, "extraction.document.fn": None, "extraction.document.f1": None}}

    pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(), runner=LocalRunner(), evaluators={"field": unknown})

    report = pipeline.evaluate(raw, document_id="d")

    assert report["metrics"]["field"]["extraction.document.f1"] is None
    assert report["review_required"] is True


@pytest.mark.parametrize("rows", [[], [{"inputs.id": "wrong"}], [{"inputs.id": "d"}, {"inputs.id": "d"}], [{"inputs.id": "d", "outputs.custom.diagnostic": "not a metric"}]])
def test_given_incomplete_runner_results_when_reported_then_rejected(rows: list[JsonObject], raw: JsonObject, mocker: MockerFixture) -> None:
    """Validate complete ID coverage and usable metric outputs from any runner."""
    runner = LocalRunner()
    mocker.patch.object(runner, "run", return_value=RunnerResult(rows=rows, metrics={}))
    pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(), runner=runner, evaluators={"custom": lambda **kwargs: {"metrics": {"score": 1}}})

    with pytest.raises(RuntimeError):
        pipeline.evaluate(raw, document_id="d")


def test_given_different_deployments_when_run_then_report_fingerprint_changes(raw: JsonObject, mocker: MockerFixture) -> None:
    """Include configured model deployment in reproducibility metadata."""
    client = mocker.Mock(spec=OpenAI)
    mocker.patch.object(LLMFieldEvaluator, "__call__", return_value={"metrics": {"score": 1}})
    reports = []
    for deployment in ("model-a", "model-b"):
        evaluator = LLMFieldEvaluator(client, deployment)
        pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(), runner=LocalRunner(), evaluators={"field": evaluator})
        reports.append(pipeline.evaluate(raw, document_id="d"))

    assert reports[0]["configuration_hash"] != reports[1]["configuration_hash"]


@pytest.mark.parametrize("offline", [False, True])
def test_given_multiple_segments_when_evaluated_then_markdown_matches_its_own_fields(offline: bool, raw: JsonObject) -> None:
    """Do not give an evaluator neighboring segments' source text or mismatched values."""
    raw["contents"][0]["markdown"] = "First segment: quantity 12000"
    second = deepcopy(raw["contents"][0])
    second.update(path="input2", markdown="Second segment: quantity 999")
    second["fields"]["items"]["valueArray"][0]["valueObject"]["quantity"]["valueNumber"] = 999
    raw["contents"].append(second)
    received: list[tuple[str, int]] = []

    def inspect(markdown: str, extracted_values: JsonObject) -> JsonObject:
        """Record exact source/value pairs injected by either pipeline."""
        received.append((markdown, extracted_values["items"][0]["quantity"]))
        return {"metrics": {"score": 1}}

    document: JsonObject = {"id": "d", "cu_result": raw}
    if offline:
        pipeline = OfflineEvaluationPipeline(OfflineEvaluationConfig(), runner=LocalRunner(), evaluators={"inspect": inspect})
        document["ground_truth"] = {"segments": [{"fields": {}}, {"fields": {}}]}
        report = pipeline._evaluate([document])
    else:
        pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(), runner=LocalRunner(), evaluators={"inspect": inspect})
        report = pipeline.run(raw, document_id="d")

    assert received == [("First segment: quantity 12000", 12000), ("Second segment: quantity 999", 999)]
    if offline:
        assert report["documents"] == 1
    else:
        assert report["id"] == "d"
    assert report["metrics"] == {"inspect": {"score": 1.0}}


@pytest.mark.parametrize("method", ["run", "evaluate"])
def test_given_single_online_document_when_reported_then_only_document_metrics_are_returned(method: str, raw: JsonObject, mocker: MockerFixture) -> None:
    """Keep online reporting independent of the offline aggregation builder."""
    mocker.patch.object(pipeline_module, "_build_report", side_effect=AssertionError("Offline report builder was called"))

    def field(**kwargs: object) -> JsonObject:
        """Return complete document-level field metrics and discarded diagnostics."""
        return {"metrics": {"extraction.document.tp": 4, "extraction.document.fp": 0, "extraction.document.fn": 0, "extraction.document.precision": 1.0, "extraction.document.recall": 1.0, "extraction.document.f1": 1.0}, "diagnostic": 99}

    pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(), runner=LocalRunner(), evaluators={"field": field})

    report = getattr(pipeline, method)(raw, document_id="d")

    assert set(report) == {"mode", "id", "metrics", "review_required", "configuration_hash"}
    assert report["mode"] == "online"
    assert report["id"] == "d"
    assert report["review_required"] is False
    assert report["metrics"]["field"] == field()["metrics"]
    assert "diagnostic" not in json.dumps(report)


@pytest.mark.parametrize("documents", [[], [{"id": "d"}], [{"id": "a"}, {"id": "b"}]])
@pytest.mark.parametrize("entrypoint", ["run_online_pipeline", "run_pipeline"])
def test_given_online_batch_when_workflow_called_then_rejected_before_runner(documents: list[JsonObject], entrypoint: str, mocker: MockerFixture) -> None:
    """Enforce the single-record contract below the public convenience methods."""
    runner = LocalRunner()
    run = mocker.patch.object(runner, "run")
    arguments = (OnlineEvaluationConfig(), documents, {"schema": SchemaEvaluator()}, runner)

    with pytest.raises(TypeError, match="dictionary"):
        if entrypoint == "run_online_pipeline":
            pipeline_module.run_online_pipeline(*arguments)  # type: ignore[arg-type]
        else:
            pipeline_module.run_pipeline(*arguments, mode="online")

    run.assert_not_called()


@pytest.mark.parametrize(("score", "minimum", "review_required"), [
    (0.9, 0.9, False),
    (0.89, 0.9, True),
    (187 / 188, 0.995, True),
    (True, 1.0, True),
    (None, 0.9, True),
])
def test_given_online_threshold_when_reported_then_review_uses_unrounded_document_score(score: float | bool | None, minimum: float, review_required: bool, raw: JsonObject) -> None:
    """Apply inclusive document thresholds without exposing gate structures."""
    def custom(**kwargs: object) -> JsonObject:
        """Return the document score under test."""
        return {"metrics": {"score": score}}

    config = OnlineEvaluationConfig(thresholds={"custom.score": minimum})
    pipeline = OnlineEvaluationPipeline(config, runner=LocalRunner(), evaluators={"custom": custom})

    report = pipeline.evaluate(raw, document_id="d")

    assert report["review_required"] is review_required
    assert "gates" not in report
    assert "gate_status" not in report
    if score == 187 / 188:
        assert report["metrics"]["custom"]["score"] == 0.995


@pytest.mark.parametrize("threshold", ["custom.missing", "field.extraction.pooled_field_f1"])
def test_given_missing_online_threshold_metric_when_reported_then_review_required(threshold: str, raw: JsonObject) -> None:
    """Treat unavailable threshold metrics as review failures, not approval."""
    pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(thresholds={threshold: 0.9}), runner=LocalRunner(), evaluators={"custom": lambda **kwargs: {"metrics": {"score": 1}}})

    report = pipeline.evaluate(raw, document_id="d")

    assert report["review_required"] is True


@pytest.mark.parametrize("review_signal", [False, True])
def test_given_passing_online_threshold_when_reported_then_evaluator_review_signal_is_preserved(review_signal: bool, raw: JsonObject) -> None:
    """Never let successful threshold checks override a requested review."""
    pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(thresholds={"custom.score": 0.9}), runner=LocalRunner(), evaluators={"custom": lambda **kwargs: {"metrics": {"score": 1}, "review_required": review_signal}})

    report = pipeline.evaluate(raw, document_id="d")

    assert report["review_required"] is review_signal


def test_given_online_runner_aggregates_when_reported_then_only_row_scores_are_used(raw: JsonObject, mocker: MockerFixture) -> None:
    """Do not leak native aggregates or mutate unrounded caller-owned results."""
    runner = LocalRunner()
    row = {"inputs.id": "d", "outputs.custom.score": 187 / 188, "outputs.custom._review_required": False}
    original = deepcopy(row)
    mocker.patch.object(runner, "run", return_value=RunnerResult(rows=[row], metrics={"custom.score": 99}, metadata={"diagnostic": "private"}))
    pipeline = OnlineEvaluationPipeline(OnlineEvaluationConfig(), runner=runner, evaluators={"custom": lambda **kwargs: {"metrics": {"score": 1}}})

    report = pipeline.evaluate(raw, document_id="d")

    assert report["metrics"] == {"custom": {"score": 0.995}}
    assert report["review_required"] is False
    assert row == original
    assert "private" not in json.dumps(report)


def test_given_offline_document_when_reported_then_existing_report_contract_is_preserved(raw: JsonObject) -> None:
    """Keep offline document aggregation and minimum-score gates unchanged."""
    def field(**kwargs: object) -> JsonObject:
        """Return complete deterministic occurrence metrics."""
        return {"metrics": {"extraction.document.tp": 4, "extraction.document.fp": 0, "extraction.document.fn": 0, "extraction.document.precision": 1.0, "extraction.document.recall": 1.0, "extraction.document.f1": 1.0}}

    config = OfflineEvaluationConfig(thresholds={"field.extraction.pooled_field_f1": 0.95})
    pipeline = OfflineEvaluationPipeline(config, runner=LocalRunner(), evaluators={"field": field})

    report = pipeline.run([{"id": "d", "cu_result": raw, "ground_truth": {}}])

    assert set(report) == {"mode", "documents", "metrics", "document_metrics", "gates", "gate_status", "configuration_hash"}
    assert report["mode"] == "offline"
    assert report["documents"] == 1
    assert report["metrics"]["field"]["extraction.mean_document_f1"] == 1.0
    assert report["metrics"]["field"]["extraction.pooled_field_f1"] == 1.0
    assert report["document_metrics"][0]["id"] == "d"
    assert report["gates"] == {"field.extraction.pooled_field_f1": True}
    assert report["gate_status"] == "pass"