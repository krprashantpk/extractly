"""Execute typed evaluation jobs with the standard Foundry evaluate function."""

from azure.ai.evaluation import evaluate

from .base import EvaluationRunner
from .types import RunnerRequest, RunnerResult


class FoundryEvaluationRunner(EvaluationRunner):
    """Use Foundry SDK execution without implicit authentication or client setup."""

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
        return RunnerResult(rows=rows, metrics=metrics, metadata=result)