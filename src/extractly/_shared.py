"""Share Foundry execution and reporting mechanics, not evaluation workflows."""

import json
import math
from collections.abc import Iterable, Mapping
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from azure.ai.evaluation import evaluate

from .config import EvaluationConfig
from .types import EvaluationRunner, Evaluator, FoundryEvaluator, JsonObject

RESERVED_EVALUATORS = {"schema", "extraction", "text_f1", "groundedness", "field_label", "coverage"}


def custom_evaluators(evaluators: Mapping[str, Evaluator] | None, runner: EvaluationRunner | None) -> dict[str, Evaluator]:
    """Validate caller-owned extensions without replacing mandatory checks."""
    if runner is not None and not callable(runner):
        raise TypeError("runner must be callable")
    custom = dict(evaluators) if evaluators is not None else {}
    if custom.keys() & RESERVED_EVALUATORS:
        raise ValueError("Custom evaluator names must not replace built-in checks")
    if any(not isinstance(name, str) or not name.strip() or not callable(evaluator) for name, evaluator in custom.items()):
        raise TypeError("Custom evaluators require nonempty string names and callable values")
    return custom


def validate_document(document: object) -> JsonObject:
    """Require a source record with an identity and saved CU result."""
    if not isinstance(document, dict):
        raise TypeError("Every document must be a dictionary")
    if not isinstance(document.get("id"), str) or not document["id"].strip():
        raise ValueError("Every document requires a nonempty id")
    if "cu_result" not in document:
        raise ValueError("Every document requires cu_result")
    return document


def write_records(path: Path, records: Iterable[JsonObject]) -> set[str]:
    """Serialize a nonempty dataset with unique IDs, rejecting invalid JSON values."""
    ids: set[str] = set()
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            identity = record["id"]
            if identity in ids:
                raise ValueError("Every document requires a unique nonempty id")
            ids.add(identity)
            output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    if not ids:
        raise ValueError("Evaluation requires at least one document")
    return ids


def _json_safe(value: object) -> object:
    """Convert undefined SDK ratios to null and reject infinite results."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, float) and math.isinf(value):
        raise RuntimeError("Foundry returned an infinite evaluation value")
    return value


def execute(config: EvaluationConfig, data: Path, expected_ids: set[str], evaluators: dict[str, FoundryEvaluator], *, evaluation_name: str, runner: EvaluationRunner | None) -> JsonObject:
    """Delegate one job and require complete rows with usable named outputs."""
    arguments: JsonObject = {
        "data": str(data), "evaluators": evaluators, "azure_ai_project": config.project,
        "evaluation_name": evaluation_name, "fail_on_evaluator_errors": True,
    }
    result = runner(**arguments) if runner is not None else dict(evaluate(**arguments))
    if not isinstance(result, dict):
        raise RuntimeError("Foundry must return an evaluation dictionary")
    rows = result.get("rows")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Foundry returned no evaluation rows for {evaluation_name}")
    if not all(isinstance(row, dict) for row in rows):
        raise RuntimeError("Foundry evaluation rows must be objects")
    if not isinstance(result.get("metrics"), dict):
        raise RuntimeError("Foundry returned no metrics dictionary")
    returned_ids = [row.get("inputs.id", row.get("id")) for row in rows]
    if any(not isinstance(identity, str) for identity in returned_ids) or set(returned_ids) != expected_ids or len(rows) != len(expected_ids):
        raise RuntimeError(f"Foundry returned incomplete coverage for {evaluation_name}")
    for row in rows:
        for name in evaluators:
            prefix = f"outputs.{name}."
            if not any(isinstance(key, str) and key.startswith(prefix) and value is not None for key, value in row.items()):
                raise RuntimeError(f"Foundry omitted {name} results")
        if "schema" in evaluators and row.get("outputs.schema.valid") not in (0, 1):
            raise RuntimeError("Foundry schema evaluation returned no valid conformance result")
    return {key: _json_safe(value) for key, value in result.items()}


def collect_metrics(results: JsonObject) -> JsonObject:
    """Retain native metrics from each independently executed job."""
    return {key: value for result in results.values() for key, value in result["metrics"].items()}


def build_report(
    config: EvaluationConfig,
    results: JsonObject,
    metrics: JsonObject,
    *,
    workflow: str,
    document_count: int,
    review_required: bool,
    fields: JsonObject,
) -> JsonObject:
    """Add common minimum-score gates and provenance to workflow-owned results."""
    gates = {
        metric: isinstance(metrics.get(metric), (int, float)) and not isinstance(metrics.get(metric), bool) and math.isfinite(metrics[metric]) and metrics[metric] >= floor
        for metric, floor in config.thresholds.items()
    }
    report: JsonObject = {
        "mode": workflow,
        "documents": document_count,
        "metrics": metrics,
        "gates": gates,
        "foundry": results,
        "configuration_hash": sha256(config.model_dump_json().encode()).hexdigest(),
        "fields": fields,
        "foundry_sdk_version": version("azure-ai-evaluation"),
        "extractly_version": version("extractly"),
        "p_error": None,
        "review_recommendation": "review_required" if review_required or any(not passed for passed in gates.values()) else "uncalibrated",
        "gate_status": "pass" if gates and all(gates.values()) else "fail" if gates else "not_configured",
    }
    json.dumps(report, allow_nan=False)
    return report
