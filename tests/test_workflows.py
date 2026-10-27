"""Verify shared contracts without recombining online and offline workflows."""

import json
from pathlib import Path

import pytest
from azure.ai.evaluation import AzureOpenAIGrader
from pytest_mock import MockerFixture

from extractly import OfflineEvaluationConfig, OfflineEvaluator, OnlineEvaluationConfig, OnlineEvaluator
from extractly.eval_runner import EvaluationRunner, RunnerRequest, RunnerResult
from extractly.types import JsonObject


@pytest.fixture(params=["offline", "online"])
def evaluator(request: pytest.FixtureRequest, local_runner: EvaluationRunner) -> OfflineEvaluator | OnlineEvaluator:
    """Provide independently configured workflows for common boundary guarantees."""
    if request.param == "offline":
        return OfflineEvaluator(OfflineEvaluationConfig(), runner=local_runner)
    return OnlineEvaluator(OnlineEvaluationConfig(), runner=local_runner)


@pytest.mark.parametrize("threshold", [1.0, 2.0])
def test_given_schema_threshold_when_run_then_gates_explicit(threshold: float, evaluator: OfflineEvaluator | OnlineEvaluator, raw: JsonObject, record: JsonObject) -> None:
    """Apply common gate mechanics to each workflow's results."""
    evaluator.config.thresholds = {"schema.valid": threshold}

    report = evaluator.run([{"id": "d", "cu_result": raw, "ground_truth": json.loads(record["response"])}])

    assert report["gate_status"] == ("pass" if threshold == 1.0 else "fail")


def test_given_missing_metric_when_run_then_gate_fails(evaluator: OfflineEvaluator | OnlineEvaluator, raw: JsonObject, record: JsonObject) -> None:
    """Do not use success-shaped defaults for unreported metrics."""
    evaluator.config.thresholds = {"unknown.metric": 0.0}

    report = evaluator.run([{"id": "d", "cu_result": raw, "ground_truth": json.loads(record["response"])}])

    assert report["gate_status"] == "fail"


def test_given_documents_when_run_then_document_metrics_are_shared_and_private(evaluator: OfflineEvaluator | OnlineEvaluator, raw: JsonObject, record: JsonObject) -> None:
    """Expose scores for each document in both workflows without raw diagnostics."""
    def custom(*, response: str, **kwargs: object) -> JsonObject:
        """Return scalar scores alongside private diagnostic details."""
        return {"score": 0.75, "passed": False, "undefined": None, "reason": "private", "details": {"source": response}}

    evaluator.evaluators["application"] = custom

    report = evaluator.run([
        {"id": identity, "cu_result": raw, "ground_truth": json.loads(record["response"])}
        for identity in ("first", "second")
    ])

    assert [document["id"] for document in report["document_metrics"]] == ["first", "second"]
    for document in report["document_metrics"]:
        assert document["schema.valid"] == 1
        assert document["schema.violations"] == 0
        assert document["application.score"] == 0.75
        assert document["application.passed"] is False
        assert document["application.undefined"] is None
        assert not {"schema.paths", "application.reason", "application.details", "extraction.comparisons", "extraction.field_counts"} & document.keys()
    assert "private" not in json.dumps(report["document_metrics"], allow_nan=False)


def test_given_fractional_metrics_when_reported_then_rounded_after_gate_checks(evaluator: OfflineEvaluator | OnlineEvaluator, raw: JsonObject, record: JsonObject, mocker: MockerFixture) -> None:
    """Round reported scores without changing gate decisions or runner values."""
    def custom(**kwargs: object) -> JsonObject:
        """Return repeating decimals alongside integer and boolean metrics."""
        return {"score": 187 / 188, "average_count": 187 / 6, "ratio": 1 / 6, "count": 7, "passed": False, "undefined": None}

    evaluator.evaluators["application"] = custom
    evaluator.config.thresholds = {"application.score": 0.995}
    run = mocker.spy(evaluator.runner, "run")

    report = evaluator.run([{"id": "d", "cu_result": raw, "ground_truth": json.loads(record["response"])}])

    for metrics in (report["metrics"], report["document_metrics"][0]):
        assert metrics["application.score"] == 0.995
        assert metrics["application.average_count"] == 31.167
        assert metrics["application.ratio"] == 0.167
    document = report["document_metrics"][0]
    assert type(document["application.count"]) is int
    assert document["application.passed"] is False
    assert document["application.undefined"] is None
    assert report["gates"]["application.score"] is False
    assert report["gate_status"] == "fail"
    assert run.spy_return.metrics["application.score"] == 187 / 188
    assert run.spy_return.rows[0]["outputs.application.score"] == 187 / 188


@pytest.mark.parametrize("documents", [[], [{"id": ""}], [{"id": "d"}]])
def test_given_invalid_batch_when_run_then_rejected(documents: list[JsonObject], evaluator: OfflineEvaluator | OnlineEvaluator) -> None:
    """Require nonempty identified source records in both workflows."""
    with pytest.raises(ValueError):
        evaluator.run(documents)


def test_given_duplicate_ids_when_run_then_rejected(evaluator: OfflineEvaluator | OnlineEvaluator, raw: JsonObject, record: JsonObject) -> None:
    """Reject collisions before any evaluation job starts."""
    document = {"id": "d", "cu_result": raw, "ground_truth": json.loads(record["response"])}

    with pytest.raises(ValueError, match="unique"):
        evaluator.run([document, document])


@pytest.mark.parametrize("provider_result", [
    {"rows": [], "metrics": {}},
    {"rows": [{"inputs.id": "other", "outputs.schema.valid": 1}], "metrics": {}},
    {"rows": [{"inputs.id": "d", "outputs.schema.valid": 1}, {"inputs.id": "d", "outputs.schema.valid": 1}], "metrics": {}},
    {"rows": [{"inputs.id": "d", "outputs.schema.valid": None}], "metrics": {}},
    {"rows": [{"inputs.id": "d", "outputs.schema.valid": 1}]},
])
def test_given_incomplete_results_when_returned_then_rejected(provider_result: JsonObject, evaluator: OfflineEvaluator | OnlineEvaluator, raw: JsonObject, record: JsonObject, mocker: MockerFixture) -> None:
    """Reject missing, duplicate, mismatched, or unusable provider outputs."""
    mocker.patch.object(evaluator.runner, "run", return_value=RunnerResult(rows=provider_result["rows"], metrics=provider_result.get("metrics")))

    with pytest.raises(RuntimeError):
        evaluator.run([{"id": "d", "cu_result": raw, "ground_truth": json.loads(record["response"])}])


def test_given_provider_failure_when_run_then_original_exception_and_cleanup(evaluator: OfflineEvaluator | OnlineEvaluator, raw: JsonObject, record: JsonObject, mocker: MockerFixture) -> None:
    """Keep resource cleanup and provider error identity consistent."""
    files: list[Path] = []
    error = ConnectionError("synthetic provider failure")

    def failing_runner(request: RunnerRequest) -> RunnerResult:
        """Capture temporary input and propagate the original provider failure."""
        files.append(request.data)
        raise error

    mocker.patch.object(evaluator.runner, "run", side_effect=failing_runner)

    with pytest.raises(ConnectionError) as caught:
        evaluator.run([{"id": "d", "cu_result": raw, "ground_truth": json.loads(record["response"])}])

    assert caught.value is error
    assert files and not files[0].exists()


def test_given_custom_check_when_configured_then_shared_execution(raw: JsonObject, local_runner: EvaluationRunner) -> None:
    """Allow application-owned extension checks on the online API."""
    def custom(*, response: str, **kwargs: object) -> JsonObject:
        """Return a synthetic application-owned decision."""
        return {"valid": int(bool(response))}

    report = OnlineEvaluator(OnlineEvaluationConfig(), runner=local_runner, evaluators={"application": custom}).evaluate(raw, document_id="d")

    assert report["metrics"]["application.valid"] == 1


@pytest.mark.parametrize("kind", [OfflineEvaluator, OnlineEvaluator])
def test_given_custom_grader_when_constructed_then_retained(kind: type[OfflineEvaluator] | type[OnlineEvaluator], local_runner: EvaluationRunner, mocker: MockerFixture) -> None:
    """Accept non-callable SDK graders without invoking or replacing them."""
    config = OfflineEvaluationConfig() if kind is OfflineEvaluator else OnlineEvaluationConfig()
    grader = mocker.NonCallableMock(spec=AzureOpenAIGrader)
    assert not callable(grader)

    evaluator = kind(config, runner=local_runner, evaluators={"application_grader": grader})

    assert evaluator.evaluators["application_grader"] is grader
    grader.assert_not_called()


@pytest.mark.parametrize("custom", [{"": lambda **kwargs: {}}, {"check": None}, {"check": object()}, {1: lambda **kwargs: {}}])
def test_given_invalid_custom_evaluator_when_constructed_then_rejected(custom: JsonObject, local_runner: EvaluationRunner) -> None:
    """Preserve name and implementation validation after widening the alias."""
    with pytest.raises(TypeError, match="Custom evaluators"):
        OnlineEvaluator(OnlineEvaluationConfig(), runner=local_runner, evaluators=custom)


@pytest.mark.parametrize("kind", [OfflineEvaluator, OnlineEvaluator])
def test_given_reserved_custom_name_when_constructed_then_rejected(kind: type[OfflineEvaluator] | type[OnlineEvaluator], local_runner: EvaluationRunner) -> None:
    """Prevent replacement of mandatory shared and workflow-specific checks."""
    config = OfflineEvaluationConfig() if kind is OfflineEvaluator else OnlineEvaluationConfig()

    with pytest.raises(ValueError, match="built-in"):
        kind(config, runner=local_runner, evaluators={"schema": lambda **kwargs: {}})


def test_given_wrong_workflow_config_when_constructed_then_rejected(local_runner: EvaluationRunner) -> None:
    """Do not silently route or coerce the other workflow's configuration."""
    with pytest.raises(TypeError, match="OfflineEvaluationConfig"):
        OfflineEvaluator(OnlineEvaluationConfig(), runner=local_runner)
    with pytest.raises(TypeError, match="OnlineEvaluationConfig"):
        OnlineEvaluator(OfflineEvaluationConfig(), runner=local_runner)
