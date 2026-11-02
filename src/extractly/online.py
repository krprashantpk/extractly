"""Evaluate live extraction against source evidence without reference labels."""

import json
import logging
from collections.abc import Iterable, Iterator, Mapping
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import Field

from ._shared import (
    build_report,
    collect_metrics,
    custom_evaluators,
    execute,
    extraction_summary,
    validate_document,
    write_records,
)
from .config import EvaluationConfig
from .cu import CUAdapter
from .cu._pages import DOCUMENT_LEVEL_FIELDS
from .eval_runner import EvaluationRunner
from .evaluators import OnlineFieldEvaluator, OnlineStructuralEvaluator, SchemaEvaluator
from .evaluators._online import OnlineChecks, PageRequest
from .types import Evaluator, JsonObject

logger = logging.getLogger(__name__)

class OnlineEvaluationConfig(EvaluationConfig):
    """Configure reference-free schema and optional Decision API checks.

    Gold-comparison diagnostics such as text_f1 are not accepted.
    """

    max_fields_per_document: int = Field(default=2000, ge=1)
    selected_field_paths: list[str] | None = None
    document_level_fields: tuple[str, ...] = DOCUMENT_LEVEL_FIELDS
    record_field: str | None = None

def _validate_decision(result: JsonObject) -> None:
    """Require boolean review decisions from Decision API evaluation."""
    for row in result["rows"]:
        if not isinstance(row.get("outputs.extraction.review_required"), bool):
            raise RuntimeError("Decision evaluation returned no usable review decision")


class OnlineEvaluator:
    """Assess extraction without gold, independently of offline scoring.

    Decision clients, custom evaluators, and the required runner remain caller-owned.
    The package never creates or closes clients or persists reports.
    """

    def __init__(
        self,
        config: OnlineEvaluationConfig,
        *,
        runner: EvaluationRunner,
        evaluators: Mapping[str, Evaluator] | None = None,
        field_evaluator: OnlineFieldEvaluator | None = None,
        structural_evaluator: OnlineStructuralEvaluator | None = None,
    ) -> None:
        """Configure reference-free checks without creating clients or service calls.

        Args:
            config: Online schema, Decision API budget, and gate policy.
            runner: Required caller-configured execution runner, such as FoundryEvaluationRunner.
            evaluators: Additional caller-configured callable checks or AzureOpenAIGrader objects.
            field_evaluator: Optional caller-configured Decision field judgments.
            structural_evaluator: Optional caller-configured Decision structure checks.

        Raises:
            TypeError: Config or a supplied dependency violates its contract.
            ValueError: A custom evaluator uses a reserved built-in name.
        """
        if not isinstance(config, OnlineEvaluationConfig):
            raise TypeError("config must be an OnlineEvaluationConfig")
        self.config = config
        self.evaluators = custom_evaluators(evaluators, runner)
        self.runner = runner
        self.checks = OnlineChecks(config, field_evaluator, structural_evaluator) if field_evaluator is not None or structural_evaluator is not None else None
        self.page_results: dict[str, JsonObject] = {}
        self.request_previews: dict[str, list[JsonObject]] = {}

    def prepare(self, raw: JsonObject, *, analyzer_definition: JsonObject, complete_document: bool = False, table_groups: dict[str, list[list[int]]] | None = None) -> list[JsonObject]:
        """Preview exact page requests and local deferrals without making service calls.

        Args:
            raw: Raw CU analysis result retaining physical-page spans and field provenance.
            analyzer_definition: Original CU analyzer definition or fieldSchema.
            complete_document: Explicit completeness assertion; only a sole page covers global rules.
            table_groups: Actual extracted table groups keyed by page_key, using original record indices.

        Returns:
            Serializable page request previews, including local structural prechecks.
        """
        if self.checks is None:
            raise ValueError("Page preparation requires a field_evaluator or structural_evaluator")
        requests = self.checks.prepare_pages(
            raw, analyzer_definition, complete_document=complete_document, table_groups=table_groups,
            selected_field_paths=self.config.selected_field_paths, document_level_fields=self.config.document_level_fields,
            record_field=self.config.record_field, max_fields_per_document=self.config.max_fields_per_document,
        )
        return [request.preview() for request in requests]

    def evaluate(
        self,
        raw: JsonObject,
        *,
        document_id: str,
        analyzer_definition: JsonObject | None = None,
        complete_document: bool = False,
        table_groups: dict[str, list[list[int]]] | None = None,
    ) -> JsonObject:
        """Assess one CU extraction against its source evidence, never gold.

        Args:
            raw: Saved CU result dictionary.
            document_id: Nonempty source identity.
            analyzer_definition: Required for Decision checks; accepts the original analyzer or fieldSchema.
            complete_document: Caller completeness assertion, never inferred from a page range.
            table_groups: Retained extraction grouping keyed by page_key and original record indices.

        Returns:
            Run-level metrics, document_metrics with this document's ID and configured scores, gates, and provenance without gold-based metrics.

        Raises:
            TypeError: Raw CU data has an unsupported type.
            ValueError: Input, evidence budgets, or model policy is invalid.
            RuntimeError: Foundry returns incomplete or unusable results.

        Provider exceptions propagate unchanged. Semantic checks and publishing
        send evidence to configured Azure services and require caller approval.
        Results are uncalibrated and never automatically approve a document.
        """
        return self.run([{"id": document_id, "cu_result": raw, "analyzer_definition": analyzer_definition, "complete_document": complete_document, "table_groups": table_groups}])

    def run(self, documents: Iterable[JsonObject]) -> JsonObject:
        """Assess a nonempty batch of id and cu_result records without gold.

        Any ground_truth in incoming records is ignored and is never serialized
        into evaluator datasets. No offline extraction or text-F1 check is run.
        The returned document_metrics list includes schema and configured
        semantic scores by document, averaging field-row scores per document.
        Injected Decision checks add explicitly model-assisted field and
        structural metrics; unresolved counts remain None, not estimated zeros.
        """
        config = OnlineEvaluationConfig.model_validate(self.config.model_dump())
        checks = OnlineChecks(config, self.checks.field, self.checks.structural) if self.checks else None
        adapter = CUAdapter(config)
        selected: dict[str, Evaluator] = {"schema": SchemaEvaluator(config), **self.evaluators}
        prepared: dict[str, list[PageRequest]] = {}
        decision_results: dict[str, JsonObject] = {}
        self.page_results = {}
        self.request_previews = {}
        logger.info("Starting online evaluation '%s'", config.name)
        with TemporaryDirectory(prefix="extractly-online-") as directory:
            data = Path(directory) / "documents.jsonl"

            def records() -> Iterator[JsonObject]:
                """Prepare evidence-bearing records with no reference-label slot."""
                for item in documents:
                    document = validate_document(item)
                    record = adapter.prepare(document["cu_result"], document["id"], enforce_context_budget=checks is None)
                    if checks and not any(text.strip() for text in record["markdown"].values()):
                        raise ValueError("Decision evaluation requires actual document Markdown evidence")
                    record["contract"] = config.model_dump_json(include={"extraction_model", "tables"})
                    if checks:
                        definition = document.get("analyzer_definition")
                        if not isinstance(definition, dict):
                            raise TypeError("Decision evaluation requires an analyzer_definition dictionary")
                        prepared[document["id"]] = checks.prepare_pages(
                            document["cu_result"], definition,
                            complete_document=document.get("complete_document", False), table_groups=document.get("table_groups"),
                            selected_field_paths=config.selected_field_paths, document_level_fields=config.document_level_fields,
                            record_field=config.record_field, max_fields_per_document=config.max_fields_per_document,
                        )
                        self.request_previews[document["id"]] = [request.preview() for request in prepared[document["id"]]]
                    yield record

            ids = write_records(data, records())
            if checks:
                self.page_results = {identity: {"pages": [checks.page_report(request, {}, status="not_run" if request.questions else "not_evaluated") for request in requests], "evaluation_status": "not_run", "pages_prepared": len(requests), "pages_completed": 0} for identity, requests in prepared.items()}
                for identity, requests in prepared.items():
                    decision_results[identity] = checks.evaluate_pages(requests, progress=self.page_results[identity])
            jobs = [("documents", data, ids, selected)]
            results: JsonObject = {}
            for name, path, expected_ids, job_evaluators in jobs:
                result = execute(config, path, expected_ids, job_evaluators, evaluation_name=f"{config.name}-online-{name}", runner=self.runner)
                if checks and name == "documents":
                    for row in result["rows"]:
                        decision = decision_results[row.get("inputs.id", row.get("id"))]
                        row.update({f"outputs.extraction.{key}": value for key, value in decision.items() if key not in ("schema_valid", "schema_violations")})
                        row["outputs.schema.valid"] = int(bool(row["outputs.schema.valid"]) and bool(decision["schema_valid"]))
                        row["outputs.schema.violations"] = row.get("outputs.schema.violations", 0) + decision["schema_violations"]
                    for key in ("schema.valid", "schema.violations", *[f"extraction.{key}" for key in decision_results[next(iter(decision_results))] if key not in ("review_required", "schema_valid", "schema_violations")]):
                        values = [row.get(f"outputs.{key}") for row in result["rows"]]
                        result["metrics"][key] = sum(values) / len(values) if all(value is not None for value in values) else None
                if checks:
                    _validate_decision(result)
                results[name] = result
        document_reviews = {row.get("inputs.id", row.get("id")): bool(not row["outputs.schema.valid"] or (checks is not None and row["outputs.extraction.review_required"])) for row in results["documents"]["rows"]}
        metrics = collect_metrics(results)
        rows = results["documents"]["rows"]
        if checks:
            for name in ("tp", "fp", "fn", "precision", "recall", "f1", *[f"{level}_{key}" for level in ("table_row", "table_count") for key in ("tp", "fp", "fn", "precision", "recall", "f1")]):
                if any(row.get(f"outputs.extraction.{name}") is None for row in rows):
                    metric = f"extraction.mean_document_{name}" if name in ("precision", "recall", "f1") else f"extraction.{name}"
                    metrics[metric] = None
            metrics.update(extraction_summary(rows))
        report = build_report(config, metrics, workflow="online", document_count=len(ids), document_reviews=document_reviews, fields={}, results=results)
        if checks:
            report["metric_basis"] = "model_assisted_page_evaluation"
            policy = {identity: {"analyzer_schema": requests[0].context["analyzer_schema"], "requests": [{key: value for key, value in request.preview().items() if key != "state"} | {"complete_document": request.context["is_complete_document"], "table_groups": request.state["table_groups"]} for request in requests]} for identity, requests in prepared.items()}
            report["configuration_hash"] = sha256(json.dumps({"evaluation": report["configuration_hash"], "decision": checks.configuration(), "pages": policy}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            report["judgments"] = [{"id": identity, **details} for identity, details in self.page_results.items()]
            report["coverage"] = {"pages_prepared": sum(details["pages_prepared"] for details in self.page_results.values()), "pages_completed": sum(details["pages_completed"] for details in self.page_results.values()), "documents": [{"id": identity, **{key: value for key, value in details.items() if key != "pages"}} for identity, details in self.page_results.items()], "unavailable_metrics": [name for name, value in report["metrics"].items() if value is None]}
            if set(config.thresholds) - set(metrics):
                raise ValueError("A gate refers to an unavailable metric name")
            report["gates"] = {name: None if metrics[name] is None else metrics[name] >= limit for name, limit in config.thresholds.items()}
            report["gate_status"] = "fail" if any(value is False for value in report["gates"].values()) else "incomplete" if any(value is None for value in report["gates"].values()) else "pass"
        logger.info("Completed online evaluation '%s' of %d documents with gate status '%s'", config.name, len(ids), report["gate_status"])
        return report
