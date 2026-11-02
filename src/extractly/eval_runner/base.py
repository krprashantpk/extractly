"""Define the execution interface shared by online and offline evaluation."""

from abc import ABC, abstractmethod

from .types import RunnerRequest, RunnerResult


class EvaluationRunner(ABC):
    """Execute typed jobs while leaving resources and dependencies caller-owned."""

    @abstractmethod
    def run(self, request: RunnerRequest) -> RunnerResult:
        """Execute the selected evaluators and return their results.

        Args:
            request: Job dataset, named evaluators, and explicit execution settings.

        Returns:
            One result row per input ID, aggregate metrics, and optional metadata.

        Raises:
            NotImplementedError: A concrete runner has not implemented execution.

        Implementations must finish reading the dataset before returning and
        propagate provider failures. They must not close caller-owned resources
        or silently substitute successful outputs when evaluation fails.
        """
        raise NotImplementedError