"""Test the independent gold-based workflow and local SDK execution."""

import json
from copy import deepcopy
from inspect import Parameter, signature
from pathlib import Path
from runpy import run_path

import pytest
from pytest_mock import MockerFixture

from extractly import EvaluationModel, OfflineEvaluationConfig, OfflineEvaluator
from extractly.eval_runner import EvaluationRunner, FoundryEvaluationRunner, RunnerResult
from extractly.evaluators import ExtractionEvaluator
from extractly.types import JsonObject


def test_given_offline_when_run_then_pooled_metrics_and_provenance(raw: JsonObject, record: JsonObject, offline_config: OfflineEvaluationConfig, local_runner: EvaluationRunner) -> None:
    """Return native offline results without introducing online assessment."""
    report = OfflineEvaluator(offline_config, runner=local_runner).evaluate(raw, document_id="d", ground_truth=json.loads(record["response"]))

    assert report["metrics"]["extraction.pooled_field_f1"] == 1
    assert report["mode"] == "offline"
    assert report["gate_status"] == "not_configured"
    assert report["p_error"] is None
    assert len(report["configuration_hash"]) == 64


def test_given_unequal_document_counts_when_run_then_pooled_field_not_document_mean(raw: JsonObject, record: JsonObject, offline_config: OfflineEvaluationConfig, local_runner: EvaluationRunner) -> None:
    """Pool occurrence counts rather than averaging per-document ratios."""
    empty = deepcopy(raw)
    empty["contents"][0]["fields"] = {}
    gold = json.loads(record["response"])

    report = OfflineEvaluator(offline_config, runner=local_runner).run([
        {"id": "positive", "cu_result": raw, "ground_truth": gold},
        {"id": "missing", "cu_result": empty, "ground_truth": gold},
    ])

    assert report["metrics"]["extraction.pooled_field_precision"] == 1.0
    assert report["metrics"]["extraction.pooled_field_recall"] == 0.5
    assert report["metrics"]["extraction.pooled_field_f1"] == 0.667
    assert report["metrics"]["extraction.mean_document_precision"] == 1.0
    assert report["metrics"]["extraction.mean_document_recall"] == 0.5
    assert report["metrics"]["extraction.mean_document_f1"] == 0.5
    assert not {"extraction.precision", "extraction.recall", "extraction.f1"} & report["metrics"].keys()
    assert not any("micro" in key for key in report["metrics"])
    assert report["review_recommendation"] == "review_required"
    assert [item["id"] for item in report["document_metrics"]] == ["positive", "missing"]
    assert [item["extraction.precision"] for item in report["document_metrics"]] == [1.0, None]
    assert [item["extraction.recall"] for item in report["document_metrics"]] == [1.0, 0.0]
    assert [item["extraction.f1"] for item in report["document_metrics"]] == [1.0, 0.0]
    assert [item["extraction.exact_match"] for item in report["document_metrics"]] == [1, 0]
    assert [item["schema.valid"] for item in report["document_metrics"]] == [1, 0]


@pytest.mark.parametrize(("matching_documents", "exact_match", "match_rate"), [(0, 0, 0.0), (5, 0, 0.833), (6, 1, 1.0)])
def test_given_document_matches_when_run_then_binary_exact_match_and_document_rate(raw: JsonObject, record: JsonObject, offline_config: OfflineEvaluationConfig, local_runner: EvaluationRunner, matching_documents: int, exact_match: int, match_rate: float) -> None:
    """Separate complete-run agreement from the proportion of exact documents."""
    empty = deepcopy(raw)
    empty["contents"][0]["fields"] = {}
    gold = json.loads(record["response"])
    offline_config.thresholds = {"extraction.exact_match": 1.0, "extraction.document_exact_match_rate": 0.8}
    documents = [{"id": str(index), "cu_result": raw if index < matching_documents else empty, "ground_truth": gold} for index in range(6)]

    report = OfflineEvaluator(offline_config, runner=local_runner).run(documents)

    assert report["metrics"]["extraction.exact_match"] == exact_match
    assert type(report["metrics"]["extraction.exact_match"]) is int
    assert report["metrics"]["extraction.document_exact_match_rate"] == match_rate
    assert [item["extraction.exact_match"] for item in report["document_metrics"]] == [int(index < matching_documents) for index in range(6)]
    assert report["gates"]["extraction.exact_match"] == bool(exact_match)
    assert report["gates"]["extraction.document_exact_match_rate"] == (matching_documents >= 5)


@pytest.mark.parametrize("empty", [False, True])
def test_given_document_when_evaluated_then_metrics_preserve_scores_without_raw_data(raw: JsonObject, record: JsonObject, local_runner: EvaluationRunner, empty: bool) -> None:
    """Expose document scores and undefined ratios without leaking extraction data."""
    predicted = deepcopy(raw)
    gold = json.loads(record["response"])
    if empty:
        predicted["contents"][0]["fields"] = {}
        gold["segments"][0]["fields"] = {}

    report = OfflineEvaluator(OfflineEvaluationConfig(), runner=local_runner).evaluate(predicted, document_id="d", ground_truth=gold)
    scores = report["document_metrics"][0]

    assert len(report["document_metrics"]) == 1
    assert scores["id"] == "d"
    assert scores["schema.valid"] == 1
    assert scores["schema.violations"] == 0
    assert scores["extraction.f1"] == (None if empty else 1.0)
    assert scores["extraction.fp"] == scores["extraction.fn"] == 0
    assert scores["extraction.exact_match"] == 1
    assert not any("tolerant" in key for key in scores)
    assert all(key == "id" or key.startswith(("schema.", "extraction.")) for key in scores)
    assert not {"schema.paths", "extraction.field_counts", "extraction.tolerant_field_counts", "extraction.comparisons"} & scores.keys()
    json.dumps(report, allow_nan=False)


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

    assert "foundry" not in report
    assert all(not key.startswith(("groundedness.", "coverage.", "field_label.")) for key in report["metrics"])


def test_given_text_diagnostic_when_enabled_then_offline_only(mocker: MockerFixture, raw: JsonObject, record: JsonObject, local_runner: EvaluationRunner) -> None:
    """Retain optional text F1 without conflating it with extraction F1."""
    diagnostic = mocker.patch("extractly.offline.F1ScoreEvaluator", return_value=lambda **kwargs: {"f1_score": 0.5})
    config = OfflineEvaluationConfig(text_f1=True)

    report = OfflineEvaluator(config, runner=local_runner).evaluate(raw, document_id="d", ground_truth=json.loads(record["response"]))

    diagnostic.assert_called_once()
    assert report["metrics"]["text_f1.f1_score"] == 0.5
    assert report["metrics"]["extraction.pooled_field_f1"] == 1
    assert report["document_metrics"][0]["text_f1.f1_score"] == 0.5


@pytest.mark.parametrize("bad_counts", [True, -1, None, 1.5])
def test_given_invalid_extraction_counts_when_returned_then_rejected(bad_counts: object, raw: JsonObject, record: JsonObject, local_runner: EvaluationRunner, mocker: MockerFixture) -> None:
    """Require nonnegative integer occurrence counts before pooled aggregation."""
    provider_result = {"rows": [{
        "inputs.id": "d", "outputs.schema.valid": 1, "outputs.extraction.tp": bad_counts,
        "outputs.extraction.fp": 0, "outputs.extraction.fn": 0,
    }], "metrics": {}}
    mocker.patch.object(local_runner, "run", return_value=RunnerResult(rows=provider_result["rows"], metrics={}))

    with pytest.raises(RuntimeError, match="occurrence counts"):
        OfflineEvaluator(OfflineEvaluationConfig(), runner=local_runner).evaluate(raw, document_id="d", ground_truth=json.loads(record["response"]))


@pytest.mark.sdk
def test_given_real_sdk_when_offline_then_native_execution_succeeds(raw: JsonObject, record: JsonObject, offline_config: OfflineEvaluationConfig) -> None:
    """Exercise installed Foundry execution without online model calls."""
    result = OfflineEvaluator(offline_config, runner=FoundryEvaluationRunner()).evaluate(raw, document_id="synthetic-sdk-test", ground_truth=json.loads(record["response"]))

    assert result["metrics"]["extraction.pooled_field_f1"] == 1
    assert "foundry" not in result


@pytest.mark.sdk
def test_given_undefined_ratios_when_sdk_aggregates_then_valid_json(raw: JsonObject, record: JsonObject, offline_config: OfflineEvaluationConfig) -> None:
    """Normalize SDK NaN to null without inventing perfect empty scores."""
    offline_config.extraction_model = None
    empty = deepcopy(raw)
    empty["contents"][0]["fields"] = {}
    gold = json.loads(record["response"])
    empty_gold = deepcopy(gold)
    empty_gold["segments"][0]["fields"] = {}

    result = OfflineEvaluator(offline_config, runner=FoundryEvaluationRunner()).run([
        {"id": "positive", "cu_result": raw, "ground_truth": gold},
        {"id": "no-values", "cu_result": empty, "ground_truth": empty_gold},
    ])

    json.dumps(result, allow_nan=False)
    assert result["metrics"]["extraction.pooled_field_f1"] == 1
    assert result["documents"] == 2


@pytest.fixture
def source_sample() -> JsonObject:
    """Load the repository example without executing its CLI entry point."""
    script = Path(__file__).resolve().parents[1] / "experiment" / "offline_evaluation.py"
    if not script.is_file() or not (script.parent / "samples").is_dir():
        pytest.skip("The ignored local six-source sample assets are not available")
    return run_path(str(script))


def test_given_six_source_pairs_when_loaded_then_all_rows_and_totals_are_valid(source_sample: JsonObject) -> None:
    """Cover all six original files, preserving 169 source-level detail rows."""
    annotations, documents = source_sample["load_documents"]()

    assert len(documents) == len({document["id"] for document in documents}) == 6
    assert [item["source_row_count"] for item in annotations["documents"]] == [2, 44, 55, 60, 1, 7]
    assert sum(len(segment["fields"]["lineItems"]) for document in documents for segment in document["ground_truth"]["segments"]) == 169


@pytest.mark.parametrize("sample_index", range(6))
def test_given_source_gold_when_compared_to_itself_then_exact_match(source_sample: JsonObject, sample_index: int) -> None:
    """Ensure each canonical annotation is a valid reference under the policy."""
    _, documents = source_sample["load_documents"]()
    reference = json.dumps(documents[sample_index]["ground_truth"], allow_nan=False)
    evaluator = ExtractionEvaluator(source_sample["make_config"]())

    result = evaluator(response=reference, ground_truth=reference)

    assert result["f1"] == result["exact_match"] == 1
    assert result["fp"] == result["fn"] == 0


def test_given_saved_cu_when_evaluated_locally_then_real_errors_and_pooled_metrics(source_sample: JsonObject) -> None:
    """Score all saved outputs with explicit LocalRunner and retain real failures."""
    _, documents = source_sample["load_documents"]()

    report = source_sample["evaluate_documents"](documents)
    summary = source_sample["summarize_report"](report)
    counts = summary["counts"]

    assert report["mode"] == "offline"
    assert report["documents"] == 6
    assert summary["pooled_field_metrics"]["f1"] == round(2 * counts["tp"] / (2 * counts["tp"] + counts["fp"] + counts["fn"]), 3)
    assert report["gate_status"] == "pass"
    assert report["review_recommendation"] == "review_required"
    assert summary["fields_with_errors"]["/segments/*/fields/deliveryMode"]["fp"] == 1
    contract = source_sample["make_config"]().extraction_schema
    assert contract is not None
    assert set(report["fields"]) == set(contract.fields)


def test_given_selected_fields_when_excluded_predictions_change_then_report_unchanged(raw: JsonObject, record: JsonObject, offline_config: OfflineEvaluationConfig, local_runner: EvaluationRunner, quantity_schema: type[EvaluationModel]) -> None:
    """Ignore excluded errors in pooled metrics, gates, and review decisions."""
    offline_config.extraction_model = quantity_schema
    offline_config.thresholds = {"extraction.pooled_field_f1": 1.0, "schema.valid": 1.0}
    changed = deepcopy(raw)
    for row in changed["contents"][0]["fields"]["items"]["valueArray"]:
        row["valueObject"]["date"] = {"valueString": "not-a-date"}
        row["valueObject"]["unused"] = {"valueString": "unexpected"}
    gold = json.loads(record["response"])
    evaluator = OfflineEvaluator(offline_config, runner=local_runner)

    original = evaluator.evaluate(raw, document_id="d", ground_truth=gold)
    selected = evaluator.evaluate(changed, document_id="d", ground_truth=gold)

    assert selected["metrics"] == original["metrics"]
    assert selected["fields"] == original["fields"]
    assert selected["gate_status"] == "pass"
    assert selected["review_recommendation"] == original["review_recommendation"]


def test_given_source_sample_when_exported_then_gold_excludes_predictions(source_sample: JsonObject, tmp_path: Path) -> None:
    """Keep canonical expected values separate from CU and retain source hashes."""
    annotations, documents = source_sample["load_documents"]()
    report = source_sample["evaluate_documents"](documents)

    summary = source_sample["save_artifacts"](annotations, documents, report, output_directory=tmp_path)
    exported = [json.loads(line) for line in (tmp_path / "golden_dataset.jsonl").read_text(encoding="utf-8").splitlines()]
    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))

    assert exported == [{"id": document["id"], "ground_truth": document["ground_truth"]} for document in documents]
    assert summary["annotation_status"] == provenance["annotation_status"] == "source_transcribed_pending_human_approval"
    assert all(len(item["source_sha256"]) == len(item["cu_result_sha256"]) == 64 for item in provenance["sources"])


def test_given_rail_rows_reordered_when_compared_then_positional_errors(source_sample: JsonObject) -> None:
    """Reject reordered rail values even though their printed row numbers match."""
    _, documents = source_sample["load_documents"]()
    expected = documents[0]["ground_truth"]
    reordered = deepcopy(expected)
    reordered["segments"][0]["fields"]["lineItems"].reverse()
    response, reference = json.dumps(reordered), json.dumps(expected)

    positional = ExtractionEvaluator(source_sample["make_config"]())(response=response, ground_truth=reference)

    assert positional["f1"] < 1
    assert positional["exact_match"] == 0
    assert positional["table_row_tp"] == len(expected["segments"][0]["fields"]["lineItems"])
