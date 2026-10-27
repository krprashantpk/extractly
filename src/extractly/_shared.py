"""Validate evaluation inputs, run Foundry jobs, and assemble shared reports.

Offline and online workflows choose their evaluators and scoring policies.
These helpers handle dataset serialization, result validation, and reporting.
"""

import json
import math
from collections.abc import Iterable, Mapping
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

from azure.ai.evaluation import AzureOpenAIGrader

from .config import EvaluationConfig
from .eval_runner import EvaluationRunner, RunnerRequest, RunnerResult
from .types import Evaluator, JsonObject

RESERVED_EVALUATORS = {"schema", "extraction", "text_f1", "groundedness", "field_label", "coverage"}


def custom_evaluators(evaluators: Mapping[str, Evaluator] | None, runner: EvaluationRunner) -> dict[str, Evaluator]:
    """Validate caller-supplied evaluators and runner, reserving built-in evaluator names.

    Args:
        evaluators: Optional mapping of nonempty custom names to callable evaluators or AzureOpenAIGrader objects; None adds no custom evaluators.
        runner: Required caller-configured EvaluationRunner implementation.

    Returns:
        A new dictionary containing the caller's custom evaluators without invoking them.

    Raises:
        TypeError: The runner does not implement EvaluationRunner, a custom evaluator name is invalid, or an implementation is neither callable nor an AzureOpenAIGrader.
        ValueError: A custom evaluator name is reserved for a built-in check.
    """
    if not isinstance(runner, EvaluationRunner):
        raise TypeError("runner must be an EvaluationRunner")
    custom = dict(evaluators) if evaluators is not None else {}
    if custom.keys() & RESERVED_EVALUATORS:
        raise ValueError("Custom evaluator names must not replace built-in checks")
    if any(not isinstance(name, str) or not name.strip() or not (callable(evaluator) or isinstance(evaluator, AzureOpenAIGrader)) for name, evaluator in custom.items()):
        raise TypeError("Custom evaluators require nonempty string names and callable or AzureOpenAIGrader values")
    return custom


def validate_document(document: object) -> JsonObject:
    """Validate a document's envelope without inspecting its CU payload.

    Args:
        document: Incoming dictionary with a nonblank string id and a cu_result key.

    Returns:
        The original dictionary unchanged; CU payload and workflow-specific validation remain with the caller.

    Raises:
        TypeError: The document is not a dictionary.
        ValueError: The id is missing, blank, or not a string, or the cu_result key is absent.
    """
    if not isinstance(document, dict):
        raise TypeError("Every document must be a dictionary")
    if not isinstance(document.get("id"), str) or not document["id"].strip():
        raise ValueError("Every document requires a nonempty id")
    if "cu_result" not in document:
        raise ValueError("Every document requires cu_result")
    return document


def write_records(path: Path, records: Iterable[JsonObject]) -> set[str]:
    """Write prepared records as UTF-8 JSONL and require a nonempty, unique-ID dataset.

    The destination is overwritten, and a failure can leave a partial file.
    Record envelopes must already contain validated, nonblank string IDs.

    Args:
        path: Destination file whose parent directory must already exist.
        records: Iterable of prepared records to serialize in full, one record per line.

    Returns:
        The set of document IDs written to the dataset.

    Raises:
        KeyError: A record has no id key.
        TypeError: A record contains a value that cannot be serialized as JSON.
        ValueError: The dataset is empty, an ID is duplicated, or a record contains nonfinite numbers or circular references.
        OSError: The destination cannot be opened or written.
    """
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
    """Replace floating-point NaN with None and reject infinity in nested results.

    Args:
        value: Result value to inspect recursively through dictionary values and list items.

    Returns:
        Rebuilt dictionaries and lists with NaN replaced by None; other values are returned unchanged.

    Raises:
        RuntimeError: A floating-point value is positive or negative infinity.
    """
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, float) and math.isinf(value):
        raise RuntimeError("Foundry returned an infinite evaluation value")
    return value


def execute(config: EvaluationConfig, data: Path, expected_ids: set[str], evaluators: dict[str, Evaluator], *, evaluation_name: str, runner: EvaluationRunner) -> JsonObject:
    """Run one evaluation job and validate its returned rows and metrics.

    Each expected ID must appear exactly once, and every row must contain at
    least one non-null output per evaluator. Schema checks must return a valid
    conformance flag. Provider exceptions propagate unchanged, and supplied
    evaluators and runners remain caller-owned.

    Args:
        config: Shared configuration supplying the optional Foundry project destination.
        data: Path to the prepared JSONL dataset passed to the evaluation runner.
        expected_ids: Exact set of document or field-row IDs required in the results.
        evaluators: Named evaluator callables or graders selected by the workflow.
        evaluation_name: Name identifying this evaluation job.
        runner: Required implementation accepting a RunnerRequest and returning a RunnerResult.

    Returns:
        The validated result dictionary with nested floating-point NaN values converted to None.

    Raises:
        RuntimeError: The runner returns an invalid result structure, incomplete coverage, missing evaluator outputs, an invalid schema flag, or infinite values.
    """
    request = RunnerRequest(data=data, evaluators=evaluators, azure_ai_project=config.project, evaluation_name=evaluation_name, fail_on_evaluator_errors=True)
    output = runner.run(request)
    if not isinstance(output, RunnerResult):
        raise RuntimeError("runner must return a RunnerResult")
    if not isinstance(output.metadata, dict):
        raise RuntimeError("runner metadata must be a dictionary")
    result = {**output.metadata, "rows": output.rows, "metrics": output.metrics}
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
    """Merge job metrics and label document-averaged extraction scores explicitly.

    Args:
        results: Mapping of job names to validated result dictionaries containing metrics dictionaries.

    Returns:
        A new metrics dictionary with extraction precision, recall, and f1 means named mean_document_precision, mean_document_recall, and mean_document_f1; values are not recalculated and later jobs overwrite earlier values with the same name.
    """
    names = {f"extraction.{score}": f"extraction.mean_document_{score}" for score in ("precision", "recall", "f1")}
    return {names.get(key, key): value for result in results.values() for key, value in result["metrics"].items()}


def collect_document_metrics(results: JsonObject) -> list[JsonObject]:
    """Collect document scores without retaining source data or diagnostic details.

    Args:
        results: Validated job results containing document rows and optional field or coverage rows.

    Returns:
        Document IDs and evaluator-qualified numeric, boolean, or undefined scores; field-row scores are averaged per document over non-null values.

    Raises:
        RuntimeError: A result row cannot be associated with an evaluated document.
    """
    documents = {row.get("inputs.id", row.get("id")): {"id": row.get("inputs.id", row.get("id"))} for row in results["documents"]["rows"]}
    for job_name, result in results.items():
        field_scores: dict[str, dict[str, list[int | float | None]]] = {}
        for row in result["rows"]:
            identity = row.get("inputs.document_id", row.get("inputs.id", row.get("id")))
            if identity not in documents:
                raise RuntimeError("Evaluation result row has no matching document identity")
            document = documents[identity]
            for key, value in row.items():
                if not isinstance(key, str) or not key.startswith("outputs."):
                    continue
                name = key.removeprefix("outputs.")
                if value is None or isinstance(value, (int, float)):
                    if job_name == "fields":
                        field_scores.setdefault(identity, {}).setdefault(name, []).append(value)
                    else:
                        document[name] = value
        for identity, scores in field_scores.items():
            for name, values in scores.items():
                numeric = [value for value in values if value is not None]
                documents[identity][name] = sum(numeric) / len(numeric) if numeric else None
    return list(documents.values())


def _round_metric_values(value: object) -> object:
    """Copy nested metric values, rounding floats while preserving other types.

    Args:
        value: Metric data containing dictionaries, lists, and JSON scalar values.

    Returns:
        A copy of metric collections with floats rounded to three decimal places and integers, booleans, strings, and None unchanged.
    """
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: _round_metric_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_metric_values(item) for item in value]
    return value


def build_report(
    config: EvaluationConfig,
    metrics: JsonObject,
    *,
    workflow: str,
    document_count: int,
    review_required: bool,
    fields: JsonObject,
    results: JsonObject,
) -> JsonObject:
    """Assemble a report with minimum-score gates and reproducibility metadata.

    Missing, nonnumeric, boolean, or nonfinite metric values fail configured
    gates. No configured thresholds produces a not_configured gate status.
    A workflow review flag or failed gate requires review; otherwise the
    recommendation remains uncalibrated. The caller owns report persistence.
    Report metrics are rounded to three decimal places after gate checks,
    without modifying the supplied metrics or runner results.

    Args:
        config: Shared configuration supplying metric thresholds and the configuration fingerprint.
        metrics: Workflow-computed and collected metrics used for gates before rounding a copy for the report.
        workflow: Workflow identifier, such as offline or online, stored as the report mode.
        document_count: Number of source documents evaluated by the workflow.
        review_required: Whether workflow-specific checks already require human review, independently of configured gates.
        fields: Per-field summaries copied with rounded floats into the report; online evaluation supplies an empty dictionary.
        results: Validated job results used to collect document-level scores without raw inputs or diagnostic details.

    Returns:
        A JSON-serializable report containing run metrics, document_metrics, field summaries, gates, review status, and configuration hash without raw job results.

    Raises:
        TypeError: The assembled report contains values that cannot be serialized as JSON.
        ValueError: The assembled report contains nonfinite numbers or circular references.
        PackageNotFoundError: Installed package metadata is unavailable for a reported dependency or Extractly.
    """
    gates = {
        metric: isinstance(metrics.get(metric), (int, float)) and not isinstance(metrics.get(metric), bool) and math.isfinite(metrics[metric]) and metrics[metric] >= floor
        for metric, floor in config.thresholds.items()
    }
    report: JsonObject = {
        "mode": workflow,
        "documents": document_count,
        "metrics": _round_metric_values(metrics),
        "document_metrics": _round_metric_values(collect_document_metrics(results)),
        "gates": gates,
        "configuration_hash": sha256(config.model_dump_json().encode()).hexdigest(),
        "fields": _round_metric_values(fields),
        "review_recommendation": "review_required" if review_required or any(not passed for passed in gates.values()) else "uncalibrated",
        "gate_status": "pass" if gates and all(gates.values()) else "fail" if gates else "not_configured",
    }
    json.dumps(report, allow_nan=False)
    return report
