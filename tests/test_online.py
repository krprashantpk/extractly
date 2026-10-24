"""Test reference-free evaluation independently of gold-based comparison."""

import json
from inspect import signature

import pytest
from pytest_mock import MockerFixture

from extractly import ModelSettings, OnlineEvaluationConfig, OnlineEvaluator
from extractly.eval_runner import EvaluationRunner, FoundryEvaluationRunner, RunnerRequest, RunnerResult
from extractly.types import JsonObject


def test_given_online_api_when_inspected_then_no_gold_or_mode_switch() -> None:
    """Expose a dedicated reference-free method rather than a shared dispatcher."""
    assert "ground_truth" not in signature(OnlineEvaluator.evaluate).parameters
    assert "mode" not in signature(OnlineEvaluator.run).parameters


def test_given_gold_in_batch_when_online_then_not_serialized(raw: JsonObject, local_runner: EvaluationRunner, mocker: MockerFixture) -> None:
    """Ignore batch reference labels completely, including in custom evaluator inputs."""
    inputs: list[JsonObject] = []
    original_run = local_runner.run

    def capture(request: RunnerRequest) -> RunnerResult:
        """Capture the actual evaluator dataset, not a proxy report field."""
        inputs.extend(json.loads(line) for line in request.data.read_text(encoding="utf-8").splitlines())
        return original_run(request)

    mocker.patch.object(local_runner, "run", side_effect=capture)
    report = OnlineEvaluator(OnlineEvaluationConfig(), runner=local_runner).run([{"id": "d", "cu_result": raw, "ground_truth": {"PRIVATE_REFERENCE": True}}])

    assert all("ground_truth" not in item for item in inputs)
    assert "PRIVATE_REFERENCE" not in json.dumps(inputs)
    assert "extraction.micro_f1" not in report["metrics"]
    assert report["fields"] == {}


def test_given_online_when_run_then_no_offline_comparator(mocker: MockerFixture, raw: JsonObject, local_runner: EvaluationRunner) -> None:
    """Do not instantiate gold-based scoring while assessing source evidence."""
    mocker.patch("extractly.offline.ExtractionEvaluator", side_effect=AssertionError("offline check in online workflow"))
    mocker.patch("extractly.offline.F1ScoreEvaluator", side_effect=AssertionError("offline diagnostic in online workflow"))

    report = OnlineEvaluator(OnlineEvaluationConfig(), runner=local_runner).evaluate(raw, document_id="d")

    assert report["mode"] == "online"
    assert set(report["metrics"]) == {"schema.valid", "schema.violations"}


def test_given_semantic_checks_when_run_then_independent_jobs_and_owned_credentials(mocker: MockerFixture, raw: JsonObject, local_runner: EvaluationRunner) -> None:
    """Use online graders while leaving the supplied credential open."""
    config = OnlineEvaluationConfig(model=ModelSettings(azure_endpoint="https://example.invalid", azure_deployment="judge"), groundedness=True, field_labels=True, completeness=True)
    credential = mocker.Mock()
    groundedness = mocker.patch("extractly.online.GroundednessEvaluator", return_value=lambda **kwargs: {"groundedness": 5})
    graders = mocker.patch("extractly.online.AzureOpenAILabelGrader", return_value=lambda **kwargs: {"passed": True})
    run = mocker.spy(local_runner, "run")

    result = OnlineEvaluator(config, credential=credential, runner=local_runner).evaluate(raw, document_id="d")

    assert groundedness.call_args.kwargs["credential"] is credential
    assert graders.call_count == 2
    assert graders.call_args.kwargs["credential"] is credential
    assert graders.call_args.kwargs["labels"] == ["supported", "unsupported", "conflicting", "missing", "ambiguous"]
    assert run.call_count == 3
    assert result["review_recommendation"] == "uncalibrated"
    credential.close.assert_not_called()


@pytest.mark.parametrize("passed", [False, None])
def test_given_nonpassing_label_when_run_then_review_or_error(passed: bool | None, mocker: MockerFixture, raw: JsonObject, local_runner: EvaluationRunner) -> None:
    """Do not approve ambiguous or absent service decisions."""
    config = OnlineEvaluationConfig(model=ModelSettings(azure_endpoint="https://example.invalid", azure_deployment="judge"), completeness=True)
    mocker.patch("extractly.online.AzureOpenAILabelGrader", return_value=lambda **kwargs: {"passed": passed, "reason": "synthetic"})
    evaluator = OnlineEvaluator(config, runner=local_runner)

    if passed is None:
        with pytest.raises(RuntimeError, match="usable decision"):
            evaluator.evaluate(raw, document_id="d")
    else:
        assert evaluator.evaluate(raw, document_id="d")["review_recommendation"] == "review_required"


@pytest.mark.parametrize("score", [True, None, 0, 6])
def test_given_invalid_groundedness_when_returned_then_rejected(score: object, mocker: MockerFixture, raw: JsonObject, local_runner: EvaluationRunner) -> None:
    """Require a real score in the SDK's documented range."""
    config = OnlineEvaluationConfig(model=ModelSettings(azure_endpoint="https://example.invalid", azure_deployment="judge"), groundedness=True)
    mocker.patch("extractly.online.GroundednessEvaluator", return_value=lambda **kwargs: {"groundedness": score, "reason": "synthetic"})

    with pytest.raises(RuntimeError, match="groundedness score"):
        OnlineEvaluator(config, runner=local_runner).evaluate(raw, document_id="d")


def test_given_low_groundedness_when_returned_then_review_required(mocker: MockerFixture, raw: JsonObject, local_runner: EvaluationRunner) -> None:
    """Apply online groundedness policy independently of offline gates."""
    config = OnlineEvaluationConfig(model=ModelSettings(azure_endpoint="https://example.invalid", azure_deployment="judge"), groundedness=True, groundedness_threshold=4)
    mocker.patch("extractly.online.GroundednessEvaluator", return_value=lambda **kwargs: {"groundedness": 3})

    report = OnlineEvaluator(config, runner=local_runner).evaluate(raw, document_id="d")

    assert report["review_recommendation"] == "review_required"


def test_given_empty_markdown_when_semantic_then_rejected_before_judging(raw: JsonObject, local_runner: EvaluationRunner) -> None:
    """Reject semantic assessment without actual document evidence."""
    raw["contents"][0]["markdown"] = ""
    config = OnlineEvaluationConfig(model=ModelSettings(azure_endpoint="https://example.invalid", azure_deployment="judge"), field_labels=True)

    with pytest.raises(ValueError, match="Markdown evidence"):
        OnlineEvaluator(config, runner=local_runner).evaluate(raw, document_id="d")


def test_given_field_budget_when_exceeded_then_rejected(raw: JsonObject, local_runner: EvaluationRunner) -> None:
    """Keep online field budget enforcement out of offline configuration."""
    config = OnlineEvaluationConfig(model=ModelSettings(azure_endpoint="https://example.invalid", azure_deployment="judge"), field_labels=True, max_fields_per_document=1)

    with pytest.raises(ValueError, match="field-grading budget"):
        OnlineEvaluator(config, runner=local_runner).evaluate(raw, document_id="d")


@pytest.mark.sdk
def test_given_real_sdk_when_online_schema_only_then_no_reference_metrics(raw: JsonObject) -> None:
    """Exercise real local SDK execution without labels or model resources."""
    report = OnlineEvaluator(OnlineEvaluationConfig(), runner=FoundryEvaluationRunner()).evaluate(raw, document_id="synthetic-online-sdk")

    assert report["metrics"]["schema.valid"] == 1
    assert not any(key.startswith("extraction.") for key in report["metrics"])
