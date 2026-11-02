"""Verify typed runners and the local and Foundry implementations."""

import json
from ast import parse
from inspect import Parameter, isabstract, signature
from pathlib import Path

import pytest
from azure.ai.evaluation import AzureOpenAIGrader
from openai import OpenAI
from pytest_mock import MockerFixture

from extractly import OfflineEvaluationConfig, OfflineEvaluator, OnlineEvaluationConfig, OnlineEvaluator
from extractly.eval_runner import EvaluationRunner, FoundryEvaluationRunner, LocalRunner, RunnerRequest, RunnerResult
from extractly.types import JsonObject


@pytest.fixture
def local_request(tmp_path: Path) -> RunnerRequest:
    """Write two records with extra columns for callable argument filtering."""
    data = tmp_path / "documents.jsonl"
    records = [{"id": "a", "response": "yes", "ground_truth": "yes"}, {"id": "b", "response": "no", "ground_truth": "yes"}]
    data.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")

    def score(*, response: str, ground_truth: str) -> JsonObject:
        """Return a numeric score alongside nonnumeric diagnostic outputs."""
        return {"score": int(response == ground_truth), "valid": response == ground_truth, "missing": None, "paths": [], "undefined": float("nan")}

    return RunnerRequest(data=data, evaluators={"match": score}, evaluation_name="synthetic")


def test_given_local_request_when_run_then_rows_and_numeric_means(local_request: RunnerRequest, mocker: MockerFixture) -> None:
    """Execute strict callable signatures without delegating to the Foundry SDK."""
    provider = mocker.patch("extractly.eval_runner.foundry.evaluate", side_effect=AssertionError("Foundry must not run"))

    result = LocalRunner().run(local_request)

    assert isinstance(LocalRunner(), EvaluationRunner)
    assert result.rows[0]["inputs.id"] == "a"
    assert result.rows[0]["inputs.response"] == "yes"
    assert result.rows[0]["outputs.match.score"] == 1
    assert result.rows[1]["outputs.match.score"] == 0
    assert result.rows[0]["outputs.match.paths"] == []
    assert result.metrics == {"match.score": 0.5}
    assert result.metadata == {}
    provider.assert_not_called()


def test_given_local_kwargs_evaluator_when_run_then_all_columns_passed(local_request: RunnerRequest) -> None:
    """Preserve all dataset columns for evaluators accepting keyword arguments."""
    def echo(**kwargs: object) -> JsonObject:
        """Return the supplied record without narrowing its columns."""
        return {"record": kwargs}

    request = RunnerRequest(data=local_request.data, evaluators={"echo": echo}, evaluation_name="synthetic")

    result = LocalRunner().run(request)

    assert result.rows[0]["outputs.echo.record"] == {"id": "a", "response": "yes", "ground_truth": "yes"}
    assert result.metrics == {}


def test_given_local_project_when_run_then_publishing_rejected(local_request: RunnerRequest) -> None:
    """Never silently drop an explicit Foundry publishing destination."""
    request = RunnerRequest(data=local_request.data, evaluators=local_request.evaluators, evaluation_name="synthetic", azure_ai_project="https://example.invalid/project")

    with pytest.raises(ValueError, match="does not publish"):
        LocalRunner().run(request)


def test_given_local_grader_when_run_then_foundry_runner_required(local_request: RunnerRequest, mocker: MockerFixture) -> None:
    """Reject SDK grader objects before processing any data."""
    grader = mocker.NonCallableMock(spec=AzureOpenAIGrader)
    request = RunnerRequest(data=local_request.data, evaluators={"grader": grader}, evaluation_name="synthetic")

    with pytest.raises(TypeError, match="FoundryEvaluationRunner"):
        LocalRunner().run(request)


@pytest.mark.parametrize("fail_on_errors", [True, False])
def test_given_local_evaluator_failure_when_run_then_failure_policy_honored(local_request: RunnerRequest, fail_on_errors: bool) -> None:
    """Propagate original failures or omit outputs without fabricating scores."""
    error = ConnectionError("synthetic evaluator failure")

    def fail(**kwargs: object) -> JsonObject:
        """Fail consistently to exercise the requested policy."""
        raise error

    request = RunnerRequest(data=local_request.data, evaluators={**local_request.evaluators, "failure": fail}, evaluation_name="synthetic", fail_on_evaluator_errors=fail_on_errors)

    if fail_on_errors:
        with pytest.raises(ConnectionError) as caught:
            LocalRunner().run(request)
        assert caught.value is error
    else:
        result = LocalRunner().run(request)
        assert len(result.rows) == 2
        assert all(not any(key.startswith("outputs.failure.") for key in row) for row in result.rows)
        assert result.metrics == {"match.score": 0.5}


def test_given_local_invalid_output_when_run_then_contract_error(local_request: RunnerRequest, mocker: MockerFixture) -> None:
    """Reject malformed evaluator output instead of silently omitting it."""
    malformed = mocker.Mock(return_value={1: "invalid key"})
    request = RunnerRequest(data=local_request.data, evaluators={"malformed": malformed}, evaluation_name="synthetic")

    with pytest.raises(RuntimeError, match="dictionary with string keys"):
        LocalRunner().run(request)


@pytest.mark.parametrize("text_f1", [False, True])
def test_given_explicit_local_runner_when_offline_then_no_foundry_execution(raw: JsonObject, record: JsonObject, text_f1: bool, mocker: MockerFixture) -> None:
    """Keep runner selection caller-owned, including local SDK text diagnostics."""
    runner = LocalRunner()
    provider = mocker.patch("extractly.eval_runner.foundry.evaluate", side_effect=AssertionError("Foundry must not run"))
    evaluator = OfflineEvaluator(OfflineEvaluationConfig(text_f1=text_f1), runner=runner)

    report = evaluator.evaluate(raw, document_id="d", ground_truth=json.loads(record["response"]))

    assert evaluator.runner is runner
    assert report["metrics"]["schema.valid"] == 1
    assert report["metrics"]["extraction.pooled_field_f1"] == 1
    if text_f1:
        assert report["metrics"]["text_f1.f1_score"] == 1
    provider.assert_not_called()


def test_given_runner_contract_when_inspected_then_abstract_run() -> None:
    """Require concrete runners to implement the typed execution method."""
    assert isabstract(EvaluationRunner)
    assert "run" in EvaluationRunner.__abstractmethods__
    assert issubclass(FoundryEvaluationRunner, EvaluationRunner)


@pytest.mark.parametrize("workflow", [OfflineEvaluator, OnlineEvaluator])
def test_given_workflow_when_inspected_then_runner_required(workflow: type[OfflineEvaluator] | type[OnlineEvaluator]) -> None:
    """Make execution strategy an explicit constructor dependency."""
    parameter = signature(workflow).parameters["runner"]

    assert parameter.default is Parameter.empty
    assert parameter.kind is Parameter.KEYWORD_ONLY


@pytest.mark.parametrize(("project", "fail_on_errors"), [(None, True), ("https://example.invalid/project", False)])
def test_given_foundry_request_when_run_then_sdk_arguments_and_metadata_preserved(project: str | None, fail_on_errors: bool, mocker: MockerFixture) -> None:
    """Translate the typed boundary without losing native SDK result metadata."""
    request = RunnerRequest(data=Path("documents.jsonl"), evaluators={}, evaluation_name="synthetic", azure_ai_project=project, fail_on_evaluator_errors=fail_on_errors)
    rows = [{"inputs.id": "d", "outputs.schema.valid": 1}]
    native = {"rows": rows, "metrics": {"schema.valid": 1.0}, "studio_url": "https://example.invalid/result"}
    provider = mocker.patch("extractly.eval_runner.foundry.evaluate", return_value=native)

    result = FoundryEvaluationRunner().run(request)

    provider.assert_called_once_with(data=str(request.data), evaluators=request.evaluators, azure_ai_project=project, evaluation_name="synthetic", fail_on_evaluator_errors=fail_on_errors)
    assert isinstance(result, RunnerResult)
    assert result.rows == rows
    assert result.metrics == {"schema.valid": 1.0}
    assert result.metadata == {"studio_url": "https://example.invalid/result"}
    assert native["rows"] == rows


def test_given_custom_grader_when_foundry_run_then_forwarded_unchanged(mocker: MockerFixture) -> None:
    """Let the SDK execute grader objects instead of invoking them as callables."""
    grader = mocker.NonCallableMock(spec=AzureOpenAIGrader)
    request = RunnerRequest(data=Path("documents.jsonl"), evaluators={"application_grader": grader}, evaluation_name="synthetic")
    native = {"rows": [{"inputs.id": "d", "outputs.application_grader.passed": True}], "metrics": {}}
    provider = mocker.patch("extractly.eval_runner.foundry.evaluate", return_value=native)

    result = FoundryEvaluationRunner().run(request)

    assert provider.call_args.kwargs["evaluators"]["application_grader"] is grader
    assert result.rows == native["rows"]
    grader.assert_not_called()


def test_given_foundry_client_when_run_then_results_published_without_changing_scores(mocker: MockerFixture) -> None:
    """Publish precomputed results through a caller-owned new Foundry client."""
    rows = [{"inputs.id": "d", "outputs.schema.valid": 1}]
    metrics = {"schema.valid": 1.0}
    mocker.patch("extractly.eval_runner.foundry.evaluate", return_value={"rows": rows, "metrics": metrics})
    client = mocker.Mock(spec=OpenAI)
    client.evals.create.return_value.id = "eval-123"
    client.evals.runs.create.return_value.id = "run-123"
    client.evals.runs.create.return_value.report_url = "https://example.invalid/new-foundry"
    request = RunnerRequest(data=Path("documents.jsonl"), evaluators={}, evaluation_name="synthetic", azure_ai_project="https://example.invalid/project")

    result = FoundryEvaluationRunner(client=client).run(request)

    assert result.rows == rows
    assert result.metrics == metrics
    client.evals.create.assert_called_once()
    assert client.evals.runs.create.call_args.kwargs["eval_id"] == "eval-123"
    assert result.metadata["eval_id"] == "eval-123"
    assert result.metadata["run_id"] == "run-123"
    assert result.metadata["report_url"] == "https://example.invalid/new-foundry"
    client.close.assert_not_called()


def test_given_mixed_outputs_when_published_then_only_common_scores_graded(mocker: MockerFixture) -> None:
    """Publish scalar document metrics without source data or rich diagnostics."""
    rows = [
        {"inputs.id": "a", "inputs.response": "large expanded source", "outputs.schema.valid": 1, "outputs.extraction.f1": 0.75, "outputs.extraction.tp": 3, "outputs.extraction.fp": 0, "outputs.optional.score": float("nan"), "outputs.flag.passed": True, "outputs.schema.details": ["diagnostic"], "outputs.extraction.comparison": {"fields": []}, "outputs.extraction.reason": "diagnostic", "outputs.optional.missing": None},
        {"inputs.id": "b", "outputs.schema.valid": 0, "outputs.extraction.f1": 0.5, "outputs.extraction.tp": 2, "outputs.extraction.fp": 1},
    ]
    metrics = {"schema.valid": 0.5, "extraction.f1": 0.625, "extraction.tp": 2.5, "extraction.fp": 0.5, "optional.score": 1, "flag.passed": 1}
    mocker.patch("extractly.eval_runner.foundry.evaluate", return_value={"rows": rows, "metrics": metrics})
    client = mocker.Mock(spec=OpenAI)
    request = RunnerRequest(data=Path("documents.jsonl"), evaluators={}, evaluation_name="synthetic")

    result = FoundryEvaluationRunner(client=client).run(request)

    criteria = client.evals.create.call_args.kwargs["testing_criteria"]
    assert [criterion["name"] for criterion in criteria] == ["schema.valid", "extraction.f1"]
    assert all(criterion["type"] == "python" for criterion in criteria)
    assert all(parse(criterion["source"]).body[0].name == "grade" for criterion in criteria)
    assert client.evals.create.call_args.kwargs["metadata"]["score_origin"] == "extractly_precomputed"
    content = client.evals.runs.create.call_args.kwargs["data_source"]["source"]["content"]
    assert content[0]["item"] == {"id": "a", "schema.valid": 1, "extraction.f1": 0.75, "extraction.tp": 3, "extraction.fp": 0, "optional.score": None, "flag.passed": True, "optional.missing": None}
    assert content[1]["item"] == {"id": "b", "schema.valid": 0, "extraction.f1": 0.5, "extraction.tp": 2, "extraction.fp": 1}
    assert "item['schema.valid']" in criteria[0]["source"]
    assert client.evals.create.call_args.kwargs["name"] == request.evaluation_name
    assert result.rows is rows
    assert result.metrics is metrics


@pytest.mark.parametrize("operation", ["create", "runs.create"])
def test_given_foundry_publishing_failure_when_run_then_original_exception(operation: str, mocker: MockerFixture) -> None:
    """Never return success when new Foundry publishing fails."""
    mocker.patch("extractly.eval_runner.foundry.evaluate", return_value={"rows": [{"outputs.schema.valid": 1}], "metrics": {"schema.valid": 1}})
    client = mocker.Mock(spec=OpenAI)
    error = ConnectionError("synthetic publishing failure")
    target = client.evals.create if operation == "create" else client.evals.runs.create
    target.side_effect = error
    request = RunnerRequest(data=Path("documents.jsonl"), evaluators={}, evaluation_name="synthetic")

    with pytest.raises(ConnectionError) as caught:
        FoundryEvaluationRunner(client=client).run(request)

    assert caught.value is error
    client.close.assert_not_called()


@pytest.mark.parametrize("rows", [[], [{"outputs.count.total": 10}]])
def test_given_no_publishable_scores_when_run_then_no_remote_evaluation_created(rows: list[JsonObject], mocker: MockerFixture) -> None:
    """Reject empty and count-only results before creating cloud resources."""
    mocker.patch("extractly.eval_runner.foundry.evaluate", return_value={"rows": rows, "metrics": {"count.total": 10}})
    client = mocker.Mock(spec=OpenAI)
    request = RunnerRequest(data=Path("documents.jsonl"), evaluators={}, evaluation_name="synthetic")

    with pytest.raises(ValueError, match="common finite numeric scores"):
        FoundryEvaluationRunner(client=client).run(request)

    client.evals.create.assert_not_called()


def test_given_foundry_failure_when_run_then_original_exception(mocker: MockerFixture) -> None:
    """Do not translate exceptions or invent successful provider results."""
    error = ConnectionError("synthetic provider failure")
    mocker.patch("extractly.eval_runner.foundry.evaluate", side_effect=error)
    request = RunnerRequest(data=Path("documents.jsonl"), evaluators={}, evaluation_name="synthetic")

    with pytest.raises(ConnectionError) as caught:
        FoundryEvaluationRunner().run(request)

    assert caught.value is error


@pytest.mark.parametrize("workflow", [OfflineEvaluator, OnlineEvaluator])
@pytest.mark.parametrize("runner", [None, lambda request: RunnerResult(rows=[], metrics={})])
def test_given_nonrunner_when_constructed_then_rejected(workflow: type[OfflineEvaluator] | type[OnlineEvaluator], runner: EvaluationRunner) -> None:
    """Reject None and legacy functions rather than silently choosing Foundry."""
    config = OfflineEvaluationConfig() if workflow is OfflineEvaluator else OnlineEvaluationConfig()

    with pytest.raises(TypeError, match="EvaluationRunner"):
        workflow(config, runner=runner)


@pytest.mark.parametrize("native", [
    {"rows": [], "metrics": None},
    {"rows": "invalid", "metrics": {}},
    {"rows": [None], "metrics": {}},
])
def test_given_invalid_sdk_containers_when_run_then_rejected(native: JsonObject, mocker: MockerFixture) -> None:
    """Do not disguise malformed SDK results as typed successful output."""
    mocker.patch("extractly.eval_runner.foundry.evaluate", return_value=native)
    request = RunnerRequest(data=Path("documents.jsonl"), evaluators={}, evaluation_name="synthetic")

    with pytest.raises(RuntimeError):
        FoundryEvaluationRunner().run(request)


def test_given_untyped_result_when_run_then_rejected(raw: JsonObject, local_runner: EvaluationRunner, mocker: MockerFixture) -> None:
    """Enforce typed results even when a custom implementation returns a dict."""
    mocker.patch.object(local_runner, "run", return_value={"rows": [], "metrics": {}})

    with pytest.raises(RuntimeError, match="RunnerResult"):
        OnlineEvaluator(OnlineEvaluationConfig(), runner=local_runner).evaluate(raw, document_id="d")


def test_given_result_metadata_when_reported_then_raw_results_omitted(raw: JsonObject, local_runner: EvaluationRunner, mocker: MockerFixture) -> None:
    """Omit raw job results while retaining metrics from typed runner outputs."""
    original_run = local_runner.run

    def run_with_metadata(request: RunnerRequest) -> RunnerResult:
        """Attach native metadata and conflicting reserved keys to a valid result."""
        result = original_run(request)
        return RunnerResult(rows=result.rows, metrics=result.metrics, metadata={"studio_url": "synthetic", "rows": [], "metrics": {}})

    mocker.patch.object(local_runner, "run", side_effect=run_with_metadata)

    report = OnlineEvaluator(OnlineEvaluationConfig(), runner=local_runner).evaluate(raw, document_id="d")

    assert "foundry" not in report
    assert "studio_url" not in report
    assert report["metrics"]["schema.valid"] == 1