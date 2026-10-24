"""Evaluate extraction against independent gold without online grading."""

import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from tempfile import TemporaryDirectory

from azure.ai.evaluation import F1ScoreEvaluator

from ._shared import build_report, collect_metrics, custom_evaluators, execute, validate_document, write_records
from .config import EvaluationConfig
from .cu import CUAdapter
from .eval_runner import EvaluationRunner
from .evaluators import ExtractionEvaluator, SchemaEvaluator, ratios
from .types import Evaluator, JsonObject

logger = logging.getLogger(__name__)


class OfflineEvaluationConfig(EvaluationConfig):
    """Configure gold-based comparison with optional text-overlap diagnostics.

    Model, credential, and semantic-grading settings are not accepted.
    """

    text_f1: bool = False


def _validate_extraction(result: JsonObject) -> None:
    """Require directional counts before computing pooled extraction metrics."""
    for row in result["rows"]:
        for key in ("tp", "fp", "fn"):
            count = row.get(f"outputs.extraction.{key}")
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise RuntimeError("Foundry returned invalid extraction occurrence counts")
        field_counts = row.get("outputs.extraction.field_counts")
        if not isinstance(field_counts, dict) or any(not isinstance(count, list) or len(count) != 3 or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in count) for count in field_counts.values()):
            raise RuntimeError("Foundry returned invalid per-field occurrence counts")


class OfflineEvaluator:
    """Compare saved CU extraction with required independent gold annotations.

    This workflow never creates credentials or instantiates online graders.
    Custom evaluators and the required execution runner remain
    caller-owned. Report persistence is the caller's responsibility.
    """

    def __init__(self, config: OfflineEvaluationConfig, *, runner: EvaluationRunner, evaluators: Mapping[str, Evaluator] | None = None) -> None:
        """Configure gold-based scoring without authentication or service calls.

        Args:
            config: Offline comparison, schema, and minimum-score gate policy.
            runner: Required caller-configured execution runner, such as FoundryEvaluationRunner.
            evaluators: Additional caller-configured callable checks or AzureOpenAIGrader objects.

        Raises:
            TypeError: Config or a supplied dependency violates its contract.
            ValueError: A custom evaluator uses a reserved built-in name.
        """
        if not isinstance(config, OfflineEvaluationConfig):
            raise TypeError("config must be an OfflineEvaluationConfig")
        self.config = config
        self.evaluators = custom_evaluators(evaluators, runner)
        self.runner = runner

    def evaluate(self, raw: JsonObject, *, document_id: str, ground_truth: JsonObject) -> JsonObject:
        """Evaluate one extraction against required canonical source annotations.

        Args:
            raw: Saved CU result dictionary.
            document_id: Nonempty source identity.
            ground_truth: Independently annotated canonical extraction, never optional.

        Returns:
            Schema results, typed occurrence metrics, field counts, gates, and provenance.

        Raises:
            TypeError: Raw CU data has an unsupported type.
            ValueError: Input, gold, or comparison policy is invalid.
            RuntimeError: Foundry returns incomplete or unusable results.

        Provider exceptions propagate unchanged. No report is written by the
        package; the caller owns persistence and any publishing approval.
        """
        return self.run([{"id": document_id, "cu_result": raw, "ground_truth": ground_truth}])

    def run(self, documents: Iterable[JsonObject]) -> JsonObject:
        """Evaluate a nonempty batch of id, cu_result, and ground_truth records.

        Gold is required for every record. Undefined ratios remain null;
        micro metrics pool occurrence counts rather than document averages.
        """
        config = OfflineEvaluationConfig.model_validate(self.config.model_dump())
        adapter = CUAdapter(config)
        selected: dict[str, Evaluator] = {"schema": SchemaEvaluator(config), "extraction": ExtractionEvaluator(config), **self.evaluators}
        if config.text_f1:
            selected["text_f1"] = F1ScoreEvaluator()

        def records() -> Iterator[JsonObject]:
            """Prepare gold-bearing records only for offline scoring."""
            for item in documents:
                document = validate_document(item)
                gold = document.get("ground_truth")
                if not isinstance(gold, dict):
                    raise ValueError("Offline evaluation requires source-level ground_truth for every document")
                record = adapter.prepare(document["cu_result"], document["id"])
                record["ground_truth"] = json.dumps(gold, ensure_ascii=False, allow_nan=False)
                yield record

        logger.info("Starting offline evaluation '%s'", config.name)
        with TemporaryDirectory(prefix="extractly-offline-") as directory:
            data = Path(directory) / "documents.jsonl"
            ids = write_records(data, records())
            result = execute(config, data, ids, selected, evaluation_name=f"{config.name}-offline-documents", runner=self.runner)
            _validate_extraction(result)
        results = {"documents": result}
        metrics = collect_metrics(results)
        totals = [sum(row[f"outputs.extraction.{key}"] for row in result["rows"]) for key in ("tp", "fp", "fn")]
        metrics.update({f"extraction.micro_{key}": value for key, value in ratios(*totals).items()})
        field_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        for row in result["rows"]:
            for path, count in row["outputs.extraction.field_counts"].items():
                field_counts[path] = [a + b for a, b in zip(field_counts[path], count, strict=True)]
        review_required = any(not row["outputs.schema.valid"] or row["outputs.extraction.fp"] or row["outputs.extraction.fn"] for row in result["rows"])
        fields = {path: {"tp": count[0], "fp": count[1], "fn": count[2], **ratios(*count)} for path, count in field_counts.items()}
        report = build_report(config, results, metrics, workflow="offline", document_count=len(ids), review_required=review_required, fields=fields)
        logger.info("Completed offline evaluation '%s' of %d documents with gate status '%s'", config.name, len(ids), report["gate_status"])
        return report
