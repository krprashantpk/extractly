"""Coordinate caller-owned reference-based evaluators and namespaced reports."""

import logging
from collections.abc import Iterable, Mapping

from ._pipeline import run_pipeline, select_evaluators
from .config import EvaluationConfig
from .eval_runner import EvaluationRunner
from .evaluators.field import FieldEvaluator
from .evaluators.schema import SchemaEvaluator
from .evaluators.structural import StructuralEvaluator
from .types import Evaluator, JsonObject

logger = logging.getLogger(__name__)


class OfflineEvaluationConfig(EvaluationConfig):
    """Configure reference-based evaluation, context limits, and metric gates."""


class OfflineEvaluationPipeline:
    """Run field, structural, and schema checks by default without owning clients."""

    def __init__(self, config: OfflineEvaluationConfig, *, runner: EvaluationRunner, evaluators: Mapping[str, Evaluator] | None = None) -> None:
        """Accept pipeline policy and an optional replacement evaluator mapping.

        Args:
            config: Shared extraction model, context budget, and metric thresholds.
            runner: Required caller-owned execution runner.
            evaluators: Named metric-returning callables; None selects field, structural, and schema defaults.

        Raises:
            TypeError: Configuration, runner, or evaluator mapping is invalid.
            ValueError: The evaluator mapping is empty.
        """
        if not isinstance(config, OfflineEvaluationConfig):
            raise TypeError("config must be an OfflineEvaluationConfig")
        self.config = config
        self.runner = runner
        defaults: dict[str, Evaluator] = {"field": FieldEvaluator(), "structural": StructuralEvaluator(), "schema": SchemaEvaluator()}
        self.evaluators = select_evaluators(defaults if evaluators is None else evaluators, runner)

    def run(self, documents: Iterable[JsonObject]) -> JsonObject:
        """Return evaluator-namespaced document means and pooled occurrence scores.

        Each record requires id, cu_result, and ground_truth. An explicit
        evaluator mapping replaces the defaults. Every evaluator must return
        metrics; diagnostics do not enter aggregate metrics. Original inputs,
        evaluators, clients, and runner remain caller-owned.
        """
        logger.info("Starting offline pipeline '%s'", self.config.name)
        report = run_pipeline(self.config, documents, self.evaluators, self.runner, mode="offline")
        logger.info("Completed offline pipeline '%s' of %s documents", self.config.name, report["documents"])
        return report