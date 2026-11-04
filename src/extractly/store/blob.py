"""Load independently annotated golden datasets from Azure Blob Storage."""

import json
import logging

from azure.storage.blob import ContainerClient

from ..types import JsonObject
from .base import GoldenDatasetStore

logger = logging.getLogger(__name__)


class AzureBlobGoldenDatasetStore(GoldenDatasetStore):
    """Read one JSON annotation blob per document using a caller-owned client."""

    def __init__(self, container_client: ContainerClient) -> None:
        """Accept a synchronous client already scoped to the annotation container.

        Args:
            container_client: Caller-configured Azure container client; authentication, retries, timeouts, and cleanup remain caller-owned.

        Raises:
            TypeError: The supplied client is not a synchronous ContainerClient.
        """
        if not isinstance(container_client, ContainerClient):
            raise TypeError("container_client must be a synchronous ContainerClient")
        self._container_client = container_client

    def load(self, prefix: str) -> dict[str, JsonObject]:
        """Load all JSON annotation blobs beneath a directory prefix.

        Args:
            prefix: Case-sensitive dataset directory prefix, with or without a trailing slash; an empty string selects the whole container.

        Returns:
            Ground-truth JSON objects keyed by their blob path relative to the prefix, with the final .json extension removed case-insensitively; nested paths and other extensions are preserved.

        Raises:
            TypeError: The prefix is not a string.
            ValueError: No JSON blobs match, a blob is not a UTF-8 JSON object, or a derived document ID is empty or duplicated.
            RuntimeError: Azure returns an invalid blob name or a name outside the requested prefix.

        Non-JSON blobs are ignored. UTF-8 byte-order marks are accepted.
        Ground-truth objects are returned unchanged, including canonical segment
        annotations. Annotation schemas remain evaluator-owned. The complete
        dataset is held in memory; no partial dataset is returned on failure.
        Azure failures propagate unchanged, and the supplied client is never closed.
        """
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        blob_prefix = f"{prefix}/" if prefix and not prefix.endswith("/") else prefix
        logger.info("Loading golden dataset from blob prefix '%s'", blob_prefix)
        annotations: dict[str, JsonObject] = {}
        for blob in self._container_client.list_blobs(name_starts_with=blob_prefix):
            blob_name = blob.name
            if not isinstance(blob_name, str) or not blob_name.startswith(blob_prefix):
                raise RuntimeError(f"Azure returned invalid blob name '{blob_name}' for prefix '{blob_prefix}'")
            if not blob_name.lower().endswith(".json"):
                continue
            document_id = blob_name[len(blob_prefix):-5]
            if not document_id.strip():
                raise ValueError(f"Blob '{blob_name}' must have a nonempty document ID before .json")
            if document_id in annotations:
                raise ValueError(f"Duplicate document ID '{document_id}' from blob '{blob_name}'")
            try:
                payload = self._container_client.download_blob(blob_name, encoding="utf-8-sig").readall()
                annotation: object = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Blob '{blob_name}' must contain a valid UTF-8 JSON object") from error
            if not isinstance(annotation, dict):
                raise ValueError(f"Blob '{blob_name}' must contain a JSON object")
            annotations[document_id] = annotation
        if not annotations:
            raise ValueError(f"No JSON annotation blobs found under prefix '{blob_prefix}'")
        logger.info("Loaded golden dataset from blob prefix '%s' with %s documents", blob_prefix, len(annotations))
        return annotations
