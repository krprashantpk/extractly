"""Verify shared contracts without recombining online and offline workflows."""

import json
from pathlib import Path
from typing import Any

import pytest

from extractly import OfflineEvaluationConfig, OfflineEvaluator, OnlineEvaluationConfig, OnlineEvaluator
from extractly.types import EvaluationRunner, JsonObject


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
def test_given_incomplete_results_when_returned_then_rejected(provider_result: JsonObject, evaluator: OfflineEvaluator | OnlineEvaluator, raw: JsonObject, record: JsonObject) -> None:
    """Reject missing, duplicate, mismatched, or unusable provider outputs."""
    evaluator.runner = lambda **kwargs: provider_result

    with pytest.raises(RuntimeError):
        evaluator.run([{"id": "d", "cu_result": raw, "ground_truth": json.loads(record["response"])}])


def test_given_provider_failure_when_run_then_original_exception_and_cleanup(evaluator: OfflineEvaluator | OnlineEvaluator, raw: JsonObject, record: JsonObject) -> None:
    """Keep resource cleanup and provider error identity consistent."""
    files: list[Path] = []
    error = ConnectionError("synthetic provider failure")

    def failing_runner(**kwargs: Any) -> JsonObject:
        """Capture temporary input and propagate the original provider failure."""
        files.append(Path(kwargs["data"]))
        raise error

    evaluator.runner = failing_runner

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
def test_given_reserved_custom_name_when_constructed_then_rejected(kind: type[OfflineEvaluator] | type[OnlineEvaluator]) -> None:
    """Prevent replacement of mandatory shared and workflow-specific checks."""
    config = OfflineEvaluationConfig() if kind is OfflineEvaluator else OnlineEvaluationConfig()

    with pytest.raises(ValueError, match="built-in"):
        kind(config, evaluators={"schema": lambda **kwargs: {}})


def test_given_wrong_workflow_config_when_constructed_then_rejected() -> None:
    """Do not silently route or coerce the other workflow's configuration."""
    with pytest.raises(TypeError, match="OfflineEvaluationConfig"):
        OfflineEvaluator(OnlineEvaluationConfig())
    with pytest.raises(TypeError, match="OnlineEvaluationConfig"):
        OnlineEvaluator(OfflineEvaluationConfig())
