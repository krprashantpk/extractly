"""Define the storage boundary for independently annotated golden datasets."""

from abc import ABC, abstractmethod

from ..types import JsonObject


class GoldenDatasetStore(ABC):
    """Load ground-truth annotations without owning clients or running evaluation."""

    @abstractmethod
    def load(self, prefix: str) -> dict[str, JsonObject]:
        """Load a dataset of ground-truth objects keyed by document ID.

        Args:
            prefix: Dataset directory prefix; an empty string selects the storage root.

        Returns:
            Independent ground-truth objects keyed by unique, nonempty document IDs.

        Raises:
            TypeError: The prefix is not a string.
            ValueError: No annotations match, annotations are invalid, or document IDs collide.
            NotImplementedError: A concrete store has not implemented loading.

        Implementations must finish loading before returning, propagate provider
        failures, and leave authentication, retries, timeouts, and cleanup to
        caller-owned clients. Loading must not generate annotations from predictions.
        """
        raise NotImplementedError
