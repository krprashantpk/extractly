"""Coordinate source-grounded LLM evaluators with caller-owned inference clients."""

import logging
from collections.abc import Mapping

from openai import OpenAI

from ._pipeline import run_pipeline, select_evaluators
from .config import EvaluationConfig
from .eval_runner import EvaluationRunner
from .evaluators.llm_field import LLMFieldEvaluator
from .evaluators.llm_structural import LLMStructuralEvaluator
from .evaluators.schema import SchemaEvaluator
from .types import Evaluator, JsonObject

logger = logging.getLogger(__name__)


class OnlineEvaluationConfig(EvaluationConfig):
    """Configure source-grounded evaluation and document-score review thresholds."""


class OnlineEvaluationPipeline:
    """Run LLM field, LLM structural, and schema checks by default, never using gold."""

    def __init__(
        self,
        config: OnlineEvaluationConfig,
        *,
        runner: EvaluationRunner,
        evaluators: Mapping[str, Evaluator] | None = None,
        client: OpenAI | None = None,
        deployment: str | None = None,
    ) -> None:
        """Accept configured evaluators or construct defaults from a caller-owned client.

        Args:
            config: Shared extraction model, context budget, and metric thresholds.
            runner: Required caller-owned execution runner.
            evaluators: Replacement metric-returning callable mapping; None selects LLM field, LLM structural, and schema defaults.
            client: Caller-owned OpenAI client required only for the default LLM evaluators.
            deployment: Model deployment required only for the default LLM evaluators.

        Raises:
            TypeError: A dependency or configuration does not satisfy its contract.
            ValueError: Defaults lack a client or deployment, or the mapping is empty.

        Authentication, retries, timeouts, cleanup, and report persistence remain
        caller-owned. No clients or credentials are implicitly created.
        """
        if not isinstance(config, OnlineEvaluationConfig):
            raise TypeError("config must be an OnlineEvaluationConfig")
        if evaluators is None:
            if client is None or deployment is None:
                raise ValueError("Default online evaluators require client and deployment")
            evaluators = {"field": LLMFieldEvaluator(client, deployment), "structural": LLMStructuralEvaluator(client, deployment), "schema": SchemaEvaluator()}
        self.config = config
        self.runner = runner
        self.evaluators = select_evaluators(evaluators, runner)

    def evaluate(self, raw: JsonObject, *, document_id: str, analyzer_definition: JsonObject | None = None) -> JsonObject:
        """Evaluate one CU result against source Markdown and original analyzer rules."""
        return self.run(raw, document_id=document_id, analyzer_definition=analyzer_definition)

    def run(self, raw: JsonObject, *, document_id: str, analyzer_definition: JsonObject | None = None) -> JsonObject:
        """Return one document's namespaced metrics and review decision.

        Args:
            raw: One CU result dictionary; batch inputs are not supported.
            document_id: Nonempty identity for the document being evaluated.
            analyzer_definition: Original analyzer rules required by the default LLM evaluators.

        Returns:
            A report with direct document metrics and review_required, incorporating evaluator signals and failures of unrounded document-score thresholds.

        Raises:
            TypeError: The CU result is not a dictionary.
            ValueError: Document identity, source data, or context budget is invalid.

        Ground truth never enters evaluator inputs. Reports contain no batch
        aggregates or gates. Model-assisted scores are not verified accuracy.
        Inputs, clients, evaluators, runners, and report persistence remain caller-owned.
        """
        if not isinstance(raw, dict):
            raise TypeError("raw must be one CU result dictionary; batch inputs are not supported")
        logger.info("Starting online pipeline '%s' for document '%s'", self.config.name, document_id)
        document = {"id": document_id, "cu_result": raw, "analyzer_definition": analyzer_definition}
        report = run_pipeline(self.config, document, self.evaluators, self.runner, mode="online")
        logger.info("Completed online pipeline '%s' for document '%s' with review_required '%s'", self.config.name, document_id, report["review_required"])
        return report