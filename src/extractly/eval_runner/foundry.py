"""Execute SDK evaluation jobs and optionally publish precomputed Foundry runs."""

import json
import logging
from math import isfinite

from azure.ai.evaluation import evaluate
from openai import OpenAI
from openai.types.eval_create_params import TestingCriterionPython
from openai.types.evals.create_eval_jsonl_run_data_source_param import SourceFileContentContent

from ..types import JsonObject
from .base import EvaluationRunner
from .types import RunnerRequest, RunnerResult

logger = logging.getLogger(__name__)


class FoundryEvaluationRunner(EvaluationRunner):
    """Use SDK scoring with optional publishing to a caller-owned Foundry client.

    New Foundry runs import precomputed per-document scores, not cloud rescoring.
    Only metrics with finite numeric outputs in [0, 1] for every row become
    grading criteria. Dataset records contain only document IDs and scalar
    per-document metrics, not source inputs or detailed evaluator diagnostics.
    Review flags are computed after runner execution and are not published.
    Portal averages are not pooled extraction metrics; returned scores are unchanged.
    """

    def __init__(self, *, client: OpenAI | None = None) -> None:
        """Accept an optional client targeting the project's /openai/v1 endpoint.

        Args:
            client: Caller-configured synchronous OpenAI client for new Foundry publishing; authentication, timeouts, retries, and cleanup remain caller-owned.

        Raises:
            TypeError: Client is not an OpenAI client or None.
        """
        if client is not None and not isinstance(client, OpenAI):
            raise TypeError("client must be an OpenAI client or None")
        self._client = client

    def run(self, request: RunnerRequest) -> RunnerResult:
        """Forward one typed request to Foundry and preserve native result fields.

        Args:
            request: Dataset and caller-configured evaluators for one job.

        Returns:
            Foundry rows and aggregate metrics with additional SDK metadata.

        Raises:
            TypeError: Request is not a RunnerRequest.
            RuntimeError: Foundry returns unusable row or metric containers.

        SDK exceptions propagate unchanged. Dataset files, evaluators, and any
        credentials used by those evaluators remain caller-owned.
        """
        if not isinstance(request, RunnerRequest):
            raise TypeError("request must be a RunnerRequest")
        result = dict(evaluate(data=str(request.data), evaluators=dict(request.evaluators), azure_ai_project=request.azure_ai_project, evaluation_name=request.evaluation_name, fail_on_evaluator_errors=request.fail_on_evaluator_errors))
        rows = result.pop("rows", None)
        metrics = result.pop("metrics", None)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise RuntimeError("Foundry evaluation rows must be objects")
        if not isinstance(metrics, dict):
            raise RuntimeError("Foundry returned no metrics dictionary")
        if self._client is not None:
            result.update(self._publish(request, rows, metrics))
        return RunnerResult(rows=rows, metrics=metrics, metadata=result)

    def _publish(self, request: RunnerRequest, rows: list[JsonObject], metrics: JsonObject) -> JsonObject:
        """Submit precomputed scores without changing scoring or owning the client.

        Args:
            request: Original job providing the published evaluation name.
            rows: Validated SDK records containing inputs and evaluator outputs.
            metrics: Original SDK metrics identifying candidate score columns.

        Returns:
            New Foundry evaluation ID, run ID, and report URL for the submitted run.

        Raises:
            ValueError: No rows or common finite scores in [0, 1] can be published.
        """
        score_names = [name for name in metrics if name.rsplit(".", 1)[-1] not in {"tp", "fp", "fn", "violations"} and rows and all(isinstance(value := row.get(f"outputs.{name}"), (int, float)) and not isinstance(value, bool) and isfinite(value) and 0 <= value <= 1 for row in rows)]
        if not score_names:
            raise ValueError("new Foundry publishing requires rows with common finite numeric scores in [0, 1]")
        criteria: list[TestingCriterionPython] = [
            {"type": "python", "name": name, "source": f'def grade(sample: dict, item: dict) -> float:\n    """Import an Extractly score computed before this cloud run."""\n    return float(item[{name!r}])\n'}
            for name in score_names
        ]
        document_metrics = [{"id": row.get("inputs.id", row.get("id")), **{key.removeprefix("outputs."): value for key, value in row.items() if key.startswith("outputs.") and (value is None or isinstance(value, (int, float)))}} for row in rows]
        content: list[SourceFileContentContent] = [{"item": json.loads(json.dumps(document), parse_constant=lambda value: None)} for document in document_metrics]
        properties: JsonObject = {name: {"type": ["number", "boolean", "null"]} for document in document_metrics for name in document if name != "id"}
        properties["id"] = {"type": ["string", "null"]}
        assert self._client is not None
        logger.info("Publishing precomputed evaluation '%s' with %s documents to new Foundry", request.evaluation_name, len(rows))
        evaluation = self._client.evals.create(name=request.evaluation_name, data_source_config={"type": "custom", "item_schema": {"type": "object", "properties": properties, "required": ["id", *score_names]}}, testing_criteria=criteria, metadata={"score_origin": "extractly_precomputed", "aggregation": "mean_of_document_scores"})
        run = self._client.evals.runs.create(eval_id=evaluation.id, name=request.evaluation_name, data_source={"type": "jsonl", "source": {"type": "file_content", "content": content}})
        logger.info("Published precomputed evaluation '%s' as run '%s': '%s'", request.evaluation_name, run.id, run.report_url)
        return {"eval_id": evaluation.id, "run_id": run.id, "report_url": run.report_url}