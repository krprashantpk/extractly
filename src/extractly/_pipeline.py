"""Adapt metric-returning evaluators to runners and assemble namespaced reports."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict
from hashlib import sha256
from inspect import Parameter, signature
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from .config import EvaluationConfig
from .cu import parse_cu_result
from .eval_runner import EvaluationRunner, RunnerRequest, RunnerResult
from .evaluators._shared import ratios
from .types import Evaluator, JsonObject


def select_evaluators(evaluators: Mapping[str, Evaluator], runner: EvaluationRunner) -> dict[str, Evaluator]:
    """Copy and validate named caller-owned evaluator callables and the runner."""
    if not isinstance(runner, EvaluationRunner):
        raise TypeError("runner must be an EvaluationRunner")
    if not isinstance(evaluators, Mapping):
        raise TypeError("evaluators must be a mapping")
    selected = dict(evaluators)
    if not selected:
        raise ValueError("evaluators must contain at least one evaluator")
    if any(not isinstance(name, str) or not name.strip() or "." in name or not callable(evaluator) for name, evaluator in selected.items()):
        raise TypeError("evaluators require nonempty names without dots and callable values")
    return selected


def validate_metrics(result: object, name: str) -> JsonObject:
    """Require an explicit nonempty dictionary of finite scalar metrics."""
    if not isinstance(result, dict) or not isinstance(result.get("metrics"), dict) or not result["metrics"]:
        raise RuntimeError(f"Evaluator '{name}' must return a nonempty metrics dictionary")
    metrics: JsonObject = result["metrics"]
    for key, value in metrics.items():
        if not isinstance(key, str) or not key.strip() or key == "_review_required":
            raise RuntimeError(f"Evaluator '{name}' returned an invalid metric name")
        if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise RuntimeError(f"Evaluator '{name}' metric '{key}' must be finite numeric, boolean, or None")
    return metrics


def aggregate_metrics(records: list[JsonObject], *, segments: bool = False) -> JsonObject:
    """Retain defined-value means and pool complete TP/FP/FN occurrence groups.

    Segment counts are summed and their ratios recomputed for one document.
    Run-level field ratios retain mean_document names and pooled_field ratios,
    matching the earlier workflow. Other count groups retain their document
    means and expose pooled_precision, pooled_recall, and pooled_f1. Unknown
    count groups never produce success-shaped pooled scores.
    """
    result: JsonObject = {}
    for key in dict.fromkeys(key for record in records for key in record):
        values = [record.get(key) for record in records]
        numeric = [value for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
        if all(isinstance(value, bool) for value in values):
            result[key] = all(values)
        elif segments and key in ("valid", "exact_match"):
            result[key] = min(numeric) if numeric else None
        else:
            result[key] = sum(numeric) / len(numeric) if numeric else None
    prefixes = {key[:-2] for key in result if key == "tp" or key.endswith(".tp")}
    for prefix in prefixes:
        keys = [f"{prefix}{name}" for name in ("tp", "fp", "fn")]
        if not all(key in result for key in keys):
            continue
        counts = [[record.get(key) for key in keys] for record in records]
        if any(value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0) for count in counts for value in count):
            raise RuntimeError("Occurrence metrics tp, fp, and fn must be nonnegative integers or None")
        complete = all(value is not None for count in counts for value in count)
        totals = [sum(cast(int, count[index]) for count in counts) for index in range(3)] if complete else None
        pooled = ratios(*totals) if totals is not None else dict.fromkeys(("precision", "recall", "f1"))
        if segments:
            result.update(dict(zip(keys, totals if totals is not None else [None, None, None], strict=True)))
            result.update({f"{prefix}{name}": value for name, value in pooled.items()})
        elif prefix == "extraction.document.":
            for name in pooled:
                result[f"extraction.mean_document_{name}"] = result.pop(f"{prefix}{name}", None)
                result[f"extraction.pooled_field_{name}"] = pooled[name]
        else:
            result.update({f"{prefix}pooled_{name}": value for name, value in pooled.items()})
    return result


def requires_review(metrics: JsonObject) -> bool:
    """Identify unresolved metrics, failed conformance, and directional errors."""
    return any(value is None or (key.rsplit(".", 1)[-1] in ("fp", "fn", "violations", "ambiguous") and value > 0) or (key.rsplit(".", 1)[-1] == "valid" and not value) for key, value in metrics.items() if key.rsplit(".", 1)[-1] not in ("precision", "recall", "f1"))


class DocumentEvaluator:
    """Bind call-time model inputs while keeping evaluator implementations independent."""

    def __init__(self, name: str, evaluator: Evaluator, config: EvaluationConfig, *, offline: bool) -> None:
        """Retain caller-owned policy and one evaluator without creating dependencies."""
        self.name = name
        self.evaluator = evaluator
        self.config = config
        self.offline = offline
        parameters = signature(evaluator).parameters.values()
        self.accepted = None if any(parameter.kind is Parameter.VAR_KEYWORD for parameter in parameters) else {parameter.name for parameter in parameters}

    def __call__(self, *, response: str, context: str, ground_truth: str | None = None, analyzer_definition: JsonObject | None = None, **kwargs: object) -> JsonObject:
        """Evaluate positional segments with their local Markdown and expose metrics."""
        candidate = json.loads(response)
        expected = json.loads(ground_truth) if self.offline and ground_truth is not None else None
        predicted_segments = candidate["segments"]
        expected_segments = expected["segments"] if isinstance(expected, dict) and "segments" in expected else [{"fields": expected}] if expected is not None else []
        if expected is not None and not isinstance(expected, dict):
            raise ValueError("ground_truth must be a normalized or canonical dictionary")
        if self.offline and expected is not None and "segments" not in expected and len(predicted_segments) != 1:
            raise ValueError("Multiple extracted segments require canonical ground_truth with matching segment positions")
        metrics: list[JsonObject] = []
        review = False
        for index in range(max(len(predicted_segments), len(expected_segments))):
            segment = predicted_segments[index] if index < len(predicted_segments) else {"fields": {}, "markdown": ""}
            extracted = segment["fields"]
            gold = expected_segments[index]["fields"] if index < len(expected_segments) else {}
            inputs = {"extracted_values": extracted, "extraction_model": self.config.extraction_model, "markdown": segment["markdown"]}
            if self.offline:
                inputs["ground_truth"] = gold
            else:
                inputs["analyzer_definition"] = analyzer_definition
            arguments = inputs if self.accepted is None else {key: value for key, value in inputs.items() if key in self.accepted}
            output = self.evaluator(**arguments)
            metrics.append(validate_metrics(output, self.name))
            review |= bool(output.get("review_required", False))
        combined = aggregate_metrics(metrics, segments=True)
        return {**combined, "_review_required": review or requires_review(combined)}


def run_pipeline(config: EvaluationConfig, documents: Iterable[JsonObject] | JsonObject, evaluators: Mapping[str, Evaluator], runner: EvaluationRunner, *, mode: str) -> JsonObject:
    """Dispatch to the offline or online workflow without preparing inputs.

    Raises:
        ValueError: Mode is neither offline nor online.
    """
    if mode == "offline":
        if isinstance(documents, dict):
            raise TypeError("Offline evaluation requires an iterable of document dictionaries")
        return run_offline_pipeline(config, documents, evaluators, runner)
    if mode == "online":
        if not isinstance(documents, dict):
            raise TypeError("Online evaluation requires one document dictionary; batch inputs are not supported")
        return run_online_pipeline(config, documents, evaluators, runner)
    raise ValueError("mode must be offline or online")


def run_offline_pipeline(config: EvaluationConfig, documents: Iterable[JsonObject], evaluators: Mapping[str, Evaluator], runner: EvaluationRunner) -> JsonObject:
    """Evaluate parsed predictions against required caller-supplied ground truth."""
    config = type(config).model_validate(config.model_dump())
    selected = select_evaluators(evaluators, runner)
    adapted = {name: DocumentEvaluator(name, evaluator, config, offline=True) for name, evaluator in selected.items()}

    def records() -> Iterator[JsonObject]:
        """Prepare reference-based runner inputs without deriving labels from CU."""
        for document in documents:
            if not isinstance(document, dict):
                raise TypeError("Every document must be a dictionary")
            identity = document.get("id")
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError("Every document requires a unique nonempty id")
            if "cu_result" not in document:
                raise ValueError("Every document requires cu_result")
            parsed = parse_cu_result(document["cu_result"])
            context = "\n\n".join(f"[{segment.content_path}]\n{segment.markdown}" for segment in parsed.segments)
            if len(context) > config.max_context_characters:
                raise ValueError("Document exceeds configured context budget; split it explicitly, never truncate")
            if not isinstance(document.get("ground_truth"), dict):
                raise ValueError("Offline evaluation requires ground_truth for every document")
            yield {
                "id": identity,
                "response": json.dumps({"segments": [asdict(segment) for segment in parsed.segments]}, ensure_ascii=False, allow_nan=False),
                "context": context,
                "query": "Evaluate the extracted document fields and all row occurrences.",
                "string_encoding": parsed.string_encoding,
                "ground_truth": json.dumps(document["ground_truth"], allow_nan=False),
            }

    return _execute_pipeline(config, records(), selected, adapted, runner, mode="offline")


def run_online_pipeline(config: EvaluationConfig, document: JsonObject, evaluators: Mapping[str, Evaluator], runner: EvaluationRunner) -> JsonObject:
    """Evaluate exactly one source-grounded document without reference labels."""
    if not isinstance(document, dict):
        raise TypeError("Online evaluation requires one document dictionary; batch inputs are not supported")
    config = type(config).model_validate(config.model_dump())
    selected = select_evaluators(evaluators, runner)
    adapted = {name: DocumentEvaluator(name, evaluator, config, offline=False) for name, evaluator in selected.items()}
    record = _prepare_record(config, document)
    return _execute_pipeline(config, [record], selected, adapted, runner, mode="online")


def _prepare_record(config: EvaluationConfig, document: JsonObject) -> JsonObject:
    """Parse one CU document and retain only supported reference-free inputs."""
    if not isinstance(document, dict):
        raise TypeError("Every document must be a dictionary")
    identity = document.get("id")
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("Every document requires a unique nonempty id")
    if "cu_result" not in document:
        raise ValueError("Every document requires cu_result")
    parsed = parse_cu_result(document["cu_result"])
    context = "\n\n".join(f"[{segment.content_path}]\n{segment.markdown}" for segment in parsed.segments)
    if len(context) > config.max_context_characters:
        raise ValueError("Document exceeds configured context budget; split it explicitly, never truncate")
    record = {
        "id": identity,
        "response": json.dumps({"segments": [asdict(segment) for segment in parsed.segments]}, ensure_ascii=False, allow_nan=False),
        "context": context,
        "query": "Evaluate the extracted document fields and all row occurrences.",
        "string_encoding": parsed.string_encoding,
    }
    if document.get("analyzer_definition") is not None:
        record["analyzer_definition"] = document["analyzer_definition"]
    return record


def _execute_pipeline(config: EvaluationConfig, records: Iterable[JsonObject], evaluators: Mapping[str, Evaluator], adapted: Mapping[str, Evaluator], runner: EvaluationRunner, *, mode: str) -> JsonObject:
    """Execute prepared inputs with exact ID coverage and temporary-file cleanup."""
    ids: set[str] = set()
    with TemporaryDirectory(prefix=f"extractly-{mode}-") as directory:
        data = Path(directory) / "documents.jsonl"
        with data.open("w", encoding="utf-8") as output:
            for record in records:
                identity = record["id"]
                if identity in ids:
                    raise ValueError("Every document requires a unique nonempty id")
                ids.add(identity)
                output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        if not ids:
            raise ValueError("Evaluation requires at least one document")
        if mode == "online" and len(ids) != 1:
            raise ValueError("Online evaluation requires exactly one document")
        result = runner.run(RunnerRequest(data=data, evaluators=adapted, evaluation_name=f"{config.name}-{mode}", azure_ai_project=config.project))
    if mode == "online":
        return _build_online_report(config, next(iter(ids)), evaluators, result)
    return _build_report(config, ids, evaluators, result, mode=mode)


def _collect_document_metrics(ids: set[str], evaluators: Mapping[str, Evaluator], result: RunnerResult) -> list[JsonObject]:
    """Validate exact document coverage and collect evaluator metrics and signals."""
    if not isinstance(result, RunnerResult) or not isinstance(result.rows, list) or not result.rows or not isinstance(result.metrics, dict):
        raise RuntimeError("runner must return a RunnerResult with rows and metrics")
    identities = [row.get("inputs.id", row.get("id")) for row in result.rows if isinstance(row, dict)]
    if len(identities) != len(result.rows) or len(identities) != len(ids) or set(identities) != ids:
        raise RuntimeError("runner returned incomplete or duplicate document coverage")
    document_metrics: list[JsonObject] = []
    for row, identity in zip(result.rows, identities, strict=True):
        scores: JsonObject = {}
        review = False
        for name in evaluators:
            prefix = f"outputs.{name}."
            metrics = {key.removeprefix(prefix): value for key, value in row.items() if isinstance(key, str) and key.startswith(prefix) and key != f"{prefix}_review_required"}
            validate_metrics({"metrics": metrics}, name)
            scores[name] = metrics
            review |= bool(row.get(f"{prefix}_review_required", False)) or requires_review(metrics)
        document_metrics.append({"id": identity, "metrics": scores, "review_required": review})
    return document_metrics


def _build_report(config: EvaluationConfig, ids: set[str], evaluators: Mapping[str, Evaluator], result: RunnerResult, *, mode: str) -> JsonObject:
    """Aggregate offline metrics and gates without mutating runner outputs."""
    document_metrics = _collect_document_metrics(ids, evaluators, result)
    grouped = {name: [document["metrics"][name] for document in document_metrics] for name in evaluators}
    aggregated = {name: aggregate_metrics(records) for name, records in grouped.items()}
    flat = {f"{name}.{key}": value for name, metrics in aggregated.items() for key, value in metrics.items()}
    gates = {key: _passes_threshold(flat.get(key), minimum) for key, minimum in config.thresholds.items()}
    report: JsonObject = {"mode": mode, "documents": len(ids), "metrics": aggregated, "document_metrics": document_metrics, "gates": gates, "gate_status": "pass" if gates and all(gates.values()) else "fail" if gates else "not_configured", "configuration_hash": _configuration_hash(config, evaluators)}
    return _round_report(report)


def _build_online_report(config: EvaluationConfig, document_id: str, evaluators: Mapping[str, Evaluator], result: RunnerResult) -> JsonObject:
    """Report direct document metrics and review signals without batch aggregation."""
    document = _collect_document_metrics({document_id}, evaluators, result)[0]
    scores = document["metrics"]
    flat = {f"{name}.{key}": value for name, metrics in scores.items() for key, value in metrics.items()}
    review_required = document["review_required"] or any(not _passes_threshold(flat.get(key), minimum) for key, minimum in config.thresholds.items())
    report: JsonObject = {"mode": "online", "id": document_id, "metrics": scores, "review_required": review_required, "configuration_hash": _configuration_hash(config, evaluators)}
    return _round_report(report)


def _passes_threshold(value: object, minimum: float) -> bool:
    """Require a defined finite numeric score at or above the configured minimum."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= minimum


def _configuration_hash(config: EvaluationConfig, evaluators: Mapping[str, Evaluator]) -> str:
    """Fingerprint configuration and evaluator deployments for either report format."""
    fingerprint = {
        "config": json.loads(config.model_dump_json()),
        "evaluators": {name: {"implementation": f"{getattr(evaluator, '__module__', type(evaluator).__module__)}.{getattr(evaluator, '__qualname__', type(evaluator).__qualname__)}", "deployment": getattr(evaluator, "deployment", None)} for name, evaluator in evaluators.items()},
    }
    return sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()


def _round_report(report: JsonObject) -> JsonObject:
    """Round report copies after threshold checks and verify JSON-safe output."""
    def rounded(value: object) -> object:
        """Round report copies after threshold checks without changing runner outputs."""
        if isinstance(value, float):
            return round(value, 3)
        if isinstance(value, dict):
            return {key: rounded(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rounded(item) for item in value]
        return value

    report = {key: rounded(value) for key, value in report.items()}
    json.dumps(report, allow_nan=False)
    return report