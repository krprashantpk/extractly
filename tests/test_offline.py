"""Test the independent gold-based workflow and local SDK execution."""

import json
from copy import deepcopy
from inspect import Parameter, signature

import pytest
from pytest_mock import MockerFixture

from extractly import OfflineEvaluationConfig, OfflineEvaluator
from extractly.types import EvaluationRunner, JsonObject


def test_given_offline_when_run_then_pooled_metrics_and_provenance(raw: JsonObject, record: JsonObject, offline_config: OfflineEvaluationConfig, local_runner: EvaluationRunner) -> None:
    """Return native offline results without introducing online assessment."""
    report = OfflineEvaluator(offline_config, runner=local_runner).evaluate(raw, document_id="d", ground_truth=json.loads(record["response"]))

    assert report["metrics"]["extraction.micro_f1"] == 1
    assert report["mode"] == "offline"
    assert report["gate_status"] == "not_configured"
    assert report["p_error"] is None
    assert len(report["configuration_hash"]) == 64


def test_given_unequal_document_counts_when_run_then_micro_not_mean(raw: JsonObject, record: JsonObject, offline_config: OfflineEvaluationConfig, local_runner: EvaluationRunner) -> None:
    """Pool occurrence counts rather than averaging per-document ratios."""
    empty = deepcopy(raw)
    empty["contents"][0]["fields"] = {}
    gold = json.loads(record["response"])

    report = OfflineEvaluator(offline_config, runner=local_runner).run([
        {"id": "positive", "cu_result": raw, "ground_truth": gold},
        {"id": "missing", "cu_result": empty, "ground_truth": gold},
    ])

    assert report["metrics"]["extraction.micro_f1"] == pytest.approx(2 / 3)
    assert report["metrics"]["extraction.f1"] == 0.5
    assert report["review_recommendation"] == "review_required"


def test_given_offline_api_when_inspected_then_gold_required_and_no_credentials() -> None:
    """Express workflow separation in the public signatures, not a mode flag."""
    assert signature(OfflineEvaluator.evaluate).parameters["ground_truth"].default is Parameter.empty
    assert "credential" not in signature(OfflineEvaluator).parameters
    assert "mode" not in signature(OfflineEvaluator.run).parameters


@pytest.mark.parametrize("gold", [None, "invalid", []])
def test_given_missing_or_invalid_gold_when_run_then_rejected(gold: object, raw: JsonObject, local_runner: EvaluationRunner) -> None:
    """Never infer online evaluation from missing reference annotations."""
    with pytest.raises(ValueError, match="ground_truth"):
        OfflineEvaluator(OfflineEvaluationConfig(), runner=local_runner).run([{"id": "d", "cu_result": raw, "ground_truth": gold}])


def test_given_offline_when_run_then_no_online_grader_instantiated(mocker: MockerFixture, raw: JsonObject, record: JsonObject, local_runner: EvaluationRunner) -> None:
    """Keep deterministic offline scoring isolated from semantic services."""
    mocker.patch("extractly.online.GroundednessEvaluator", side_effect=AssertionError("online check in offline workflow"))
    mocker.patch("extractly.online.AzureOpenAILabelGrader", side_effect=AssertionError("online grader in offline workflow"))

    report = OfflineEvaluator(OfflineEvaluationConfig(), runner=local_runner).evaluate(raw, document_id="d", ground_truth=json.loads(record["response"]))

    assert set(report["foundry"]) == {"documents"}
    assert all(not key.startswith(("groundedness.", "coverage.", "field_label.")) for key in report["metrics"])


def test_given_text_diagnostic_when_enabled_then_offline_only(mocker: MockerFixture, raw: JsonObject, record: JsonObject, local_runner: EvaluationRunner) -> None:
    """Retain optional text F1 without conflating it with extraction F1."""
    diagnostic = mocker.patch("extractly.offline.F1ScoreEvaluator", return_value=lambda **kwargs: {"f1_score": 0.5})
    config = OfflineEvaluationConfig(text_f1=True)

    report = OfflineEvaluator(config, runner=local_runner).evaluate(raw, document_id="d", ground_truth=json.loads(record["response"]))

    diagnostic.assert_called_once()
    assert report["metrics"]["text_f1.f1_score"] == 0.5
    assert report["metrics"]["extraction.micro_f1"] == 1


@pytest.mark.parametrize("bad_counts", [True, -1, None, 1.5])
def test_given_invalid_extraction_counts_when_returned_then_rejected(bad_counts: object, raw: JsonObject, record: JsonObject) -> None:
    """Require nonnegative integer occurrence counts before micro aggregation."""
    provider_result = {"rows": [{
        "inputs.id": "d", "outputs.schema.valid": 1, "outputs.extraction.tp": bad_counts,
        "outputs.extraction.fp": 0, "outputs.extraction.fn": 0,
    }], "metrics": {}}

    with pytest.raises(RuntimeError, match="occurrence counts"):
        OfflineEvaluator(OfflineEvaluationConfig(), runner=lambda **kwargs: provider_result).evaluate(raw, document_id="d", ground_truth=json.loads(record["response"]))


@pytest.mark.sdk
def test_given_real_sdk_when_offline_then_native_execution_succeeds(raw: JsonObject, record: JsonObject, offline_config: OfflineEvaluationConfig) -> None:
    """Exercise installed Foundry execution without online model calls."""
    result = OfflineEvaluator(offline_config).evaluate(raw, document_id="synthetic-sdk-test", ground_truth=json.loads(record["response"]))

    assert result["metrics"]["extraction.micro_f1"] == 1
    assert result["foundry"]["documents"]["rows"][0]["inputs.id"] == "synthetic-sdk-test"


@pytest.mark.sdk
def test_given_undefined_ratios_when_sdk_aggregates_then_valid_json(raw: JsonObject, record: JsonObject, offline_config: OfflineEvaluationConfig) -> None:
    """Normalize SDK NaN to null without inventing perfect empty scores."""
    empty = deepcopy(raw)
    empty["contents"][0]["fields"] = {}
    gold = json.loads(record["response"])
    empty_gold = deepcopy(gold)
    empty_gold["segments"][0]["fields"] = {}

    result = OfflineEvaluator(offline_config).run([
        {"id": "positive", "cu_result": raw, "ground_truth": gold},
        {"id": "no-values", "cu_result": empty, "ground_truth": empty_gold},
    ])

    json.dumps(result, allow_nan=False)
    assert result["metrics"]["extraction.micro_f1"] == 1
    assert result["foundry"]["documents"]["rows"][1]["outputs.extraction.f1"] is None
