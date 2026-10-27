"""Test reference-free evaluation independently of gold-based comparison."""

import jsonimport runpy
from inspect import signature
from pathlib import Path

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
    assert "extraction.pooled_field_f1" not in report["metrics"]
    assert "extraction.mean_document_f1" not in report["metrics"]
    assert report["fields"] == {}


def test_given_online_when_run_then_no_offline_comparator(mocker: MockerFixture, raw: JsonObject, local_runner: EvaluationRunner) -> None:
    """Do not instantiate gold-based scoring while assessing source evidence."""
    mocker.patch("extractly.offline.ExtractionEvaluator", side_effect=AssertionError("offline check in online workflow"))
    mocker.patch("extractly.offline.F1ScoreEvaluator", side_effect=AssertionError("offline diagnostic in online workflow"))

    report = OnlineEvaluator(OnlineEvaluationConfig(), runner=local_runner).evaluate(raw, document_id="d")

    assert report["mode"] == "online"
    assert set(report["metrics"]) == {"schema.valid", "schema.violations"}
    assert report["document_metrics"] == [{"id": "d", "schema.valid": 1, "schema.violations": 0}]


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
    assert "foundry" not in result
    assert result["document_metrics"] == [{
        "id": "d", "schema.valid": 1, "schema.violations": 0,
        "groundedness.groundedness": 5, "field_label.passed": 1.0, "coverage.passed": True,
    }]
    credential.close.assert_not_called()


def test_given_multiple_documents_when_semantic_jobs_run_then_metrics_stay_associated(mocker: MockerFixture, raw: JsonObject, local_runner: EvaluationRunner) -> None:
    """Average field decisions per document and merge coverage by identity."""
    def field_grade(*, document_id: str, field_path: str, **kwargs: object) -> JsonObject:
        """Return different support rates for each synthetic document."""
        return {"passed": document_id == "first:batch" or field_path.endswith("/quantity"), "score": None, "reason": "private field details"}

    def coverage_grade(**kwargs: object) -> JsonObject:
        """Return opposite coverage decisions for the two source documents."""
        return {"passed": kwargs["id"] == "second", "reason": "private coverage details"}

    config = OnlineEvaluationConfig(model=ModelSettings(azure_endpoint="https://example.invalid", azure_deployment="judge"), field_labels=True, completeness=True)
    mocker.patch("extractly.online.AzureOpenAILabelGrader", side_effect=[field_grade, coverage_grade])

    report = OnlineEvaluator(config, runner=local_runner).run([
        {"id": identity, "cu_result": raw} for identity in ("first:batch", "second")
    ])
    documents = {document["id"]: document for document in report["document_metrics"]}

    assert documents["first:batch"]["field_label.passed"] == 1.0
    assert documents["second"]["field_label.passed"] == 0.25
    assert documents["first:batch"]["coverage.passed"] is False
    assert documents["second"]["coverage.passed"] is True
    assert all(document["field_label.score"] is None for document in documents.values())
    assert "private" not in json.dumps(report["document_metrics"], allow_nan=False)


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


@pytest.fixture
def online_sample() -> JsonObject:
    """Load the sample's public helpers without invoking its CLI."""
    path = Path(__file__).resolve().parents[1] / "experiment" / "online_evaluation.py"
    if not path.is_file():
        pytest.skip("The experiment directory contains optional, Git-ignored local samples")
    return runpy.run_path(str(path))


@pytest.mark.parametrize("semantic", [False, True])
def test_given_online_sample_when_evaluated_then_six_documents_without_gold(online_sample: JsonObject, semantic: bool, local_runner: EvaluationRunner, mocker: MockerFixture) -> None:
    """Exercise real sample preparation with local schema checks or mocked judges."""
    documents = online_sample["load_documents"]()
    model = ModelSettings(azure_endpoint="https://example.invalid", azure_deployment="judge") if semantic else None
    config = online_sample["make_config"](model=model)
    mocker.patch("extractly.online.GroundednessEvaluator", return_value=lambda **kwargs: {"groundedness": 5})
    mocker.patch("extractly.online.AzureOpenAILabelGrader", return_value=lambda **kwargs: {"passed": True})
    run = mocker.spy(local_runner, "run")

    report = online_sample["evaluate_documents"](documents, config=config, runner=local_runner)

    assert len(documents) == len({document["id"] for document in documents}) == report["documents"] == 6
    assert all(set(document) == {"id", "cu_result"} for document in documents)
    assert report["mode"] == "online"
    assert report["metrics"]["schema.valid"] == 1.0
    assert len(report["document_metrics"]) == 6
    assert report["gate_status"] == "pass"
    assert report["review_recommendation"] == "uncalibrated"
    assert run.call_count == (3 if semantic else 1)
    assert config.groundedness is config.field_labels is config.completeness is semantic
    assert not any(name.startswith("extraction.") for name in report["metrics"])
    if semantic:
        assert all(row["field_label.passed"] == 1.0 and row["coverage.passed"] is True for row in report["document_metrics"])


def test_given_online_sample_when_saved_then_no_gold_evidence_or_secret(online_sample: JsonObject, tmp_path: Path, local_runner: EvaluationRunner, monkeypatch: pytest.MonkeyPatch) -> None:
    """Persist compact scores and reproducibility data without sensitive inputs."""
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "sample-test-secret")
    config = online_sample["make_config"]()
    documents = online_sample["load_documents"]()
    report = online_sample["evaluate_documents"](documents, config=config, runner=local_runner)

    summary = online_sample["save_artifacts"](documents, report, config=config, output_directory=tmp_path)

    assert {path.name for path in tmp_path.iterdir()} == {"report.json", "summary.json", "configuration.json", "provenance.json"}
    assert summary == json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["uses_ground_truth"] is False
    assert len(provenance["sources"]) == 6
    assert all(len(source["cu_result_sha256"]) == 64 for source in provenance["sources"])
    assert all("sample-test-secret" not in path.read_text(encoding="utf-8") and '"cu_result":' not in path.read_text(encoding="utf-8") for path in tmp_path.iterdir())


@pytest.mark.parametrize("invalid_json", [None, "[]"])
def test_given_missing_or_invalid_saved_results_when_sample_loads_then_rejected(online_sample: JsonObject, tmp_path: Path, invalid_json: str | None) -> None:
    """Fail clearly rather than evaluate an empty or non-object dataset."""
    if invalid_json is not None:
        (tmp_path / "invalid.json").write_text(invalid_json, encoding="utf-8")

    with pytest.raises(ValueError, match=r"No saved CU result|must be a JSON object"):
        online_sample["load_documents"](samples_directory=tmp_path)


def test_given_semantic_sample_without_key_when_cli_runs_then_configuration_error(online_sample: JsonObject, monkeypatch: pytest.MonkeyPatch, mocker: MockerFixture) -> None:
    """Reject missing credentials before constructing service-backed graders."""
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("sys.argv", ["online_evaluation.py", "--semantic"])
    grader = mocker.patch("extractly.online.AzureOpenAILabelGrader")

    with pytest.raises(SystemExit) as error:
        online_sample["main"]()

    assert error.value.code == 2
    grader.assert_not_called()
