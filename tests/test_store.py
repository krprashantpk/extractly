"""Verify golden-dataset loading without credentials or Azure service calls."""

import json
import logging
from collections.abc import Iterator
from inspect import isabstract

import pytest
from azure.core.exceptions import HttpResponseError
from azure.storage.blob import BlobProperties, ContainerClient
from pytest_mock import MockerFixture

from extractly import EvaluationModel, OfflineEvaluationConfig, OfflineEvaluationPipeline
from extractly.eval_runner import LocalRunner
from extractly.store import AzureBlobGoldenDatasetStore, GoldenDatasetStore
from extractly.types import JsonObject


def test_given_store_contract_when_inspected_then_load_is_abstract() -> None:
    """Require interchangeable stores to implement the dataset loading boundary."""
    assert isabstract(GoldenDatasetStore)
    assert GoldenDatasetStore.__abstractmethods__ == {"load"}
    assert issubclass(AzureBlobGoldenDatasetStore, GoldenDatasetStore)


@pytest.mark.parametrize(("prefix", "blob_prefix"), [("gold", "gold/"), ("gold/", "gold/"), ("", "")])
def test_given_annotation_blobs_when_loaded_then_relative_ids_and_gold_are_preserved(prefix: str, blob_prefix: str, mocker: MockerFixture, caplog: pytest.LogCaptureFixture) -> None:
    """Preserve nested IDs, independent annotations, and caller-owned resources."""
    client = mocker.MagicMock(spec=ContainerClient)
    gold = {"items": [{"name": "caf\u00e9", "quantity": 0, "reviewed": False, "note": None}]}
    canonical = {"segments": [{"fields": {"items": []}}]}
    client.list_blobs.return_value = iter([
        BlobProperties(name=f"{blob_prefix}invoice.v2.json"),
        BlobProperties(name=f"{blob_prefix}notes.txt"),
        BlobProperties(name=f"{blob_prefix}archive/delivery.JSON"),
    ])
    client.download_blob.side_effect = [
        mocker.Mock(**{"readall.return_value": json.dumps(gold, ensure_ascii=False)}),
        mocker.Mock(**{"readall.return_value": json.dumps(canonical)}),
    ]

    with caplog.at_level(logging.INFO, logger="extractly.store.blob"):
        dataset = AzureBlobGoldenDatasetStore(client).load(prefix)

    assert dataset == {"invoice.v2": gold, "archive/delivery": canonical}
    client.list_blobs.assert_called_once_with(name_starts_with=blob_prefix)
    assert client.download_blob.call_args_list == [
        mocker.call(f"{blob_prefix}invoice.v2.json", encoding="utf-8-sig"),
        mocker.call(f"{blob_prefix}archive/delivery.JSON", encoding="utf-8-sig"),
    ]
    client.close.assert_not_called()
    client.__enter__.assert_not_called()
    assert f"Loading golden dataset from blob prefix '{blob_prefix}'" in caplog.text
    assert f"Loaded golden dataset from blob prefix '{blob_prefix}' with 2 documents" in caplog.text


def test_given_invalid_client_when_constructed_then_rejected(mocker: MockerFixture) -> None:
    """Reject unsupported dependencies without attempting authentication."""
    with pytest.raises(TypeError, match="container_client must be a synchronous ContainerClient"):
        AzureBlobGoldenDatasetStore(mocker.Mock(spec=object))


@pytest.mark.parametrize("prefix", [None, 42, False, b"gold/"])
def test_given_invalid_prefix_when_loaded_then_no_storage_calls(prefix: object, mocker: MockerFixture) -> None:
    """Validate operation inputs before making any provider request."""
    client = mocker.Mock(spec=ContainerClient)

    with pytest.raises(TypeError, match="prefix must be a string"):
        AzureBlobGoldenDatasetStore(client).load(prefix)  # type: ignore[arg-type]

    client.list_blobs.assert_not_called()
    client.download_blob.assert_not_called()


@pytest.mark.parametrize("blob_names", [[], ["gold/notes.txt", "gold/document.pdf"]])
def test_given_no_json_annotations_when_loaded_then_empty_dataset_is_rejected(blob_names: list[str], mocker: MockerFixture) -> None:
    """Do not disguise a missing or misconfigured dataset as successful loading."""
    client = mocker.Mock(spec=ContainerClient)
    client.list_blobs.return_value = [BlobProperties(name=name) for name in blob_names]

    with pytest.raises(ValueError, match="No JSON annotation blobs found under prefix 'gold/'"):
        AzureBlobGoldenDatasetStore(client).load("gold")

    client.download_blob.assert_not_called()


@pytest.mark.parametrize("payload", ["{", "[]", "null", "42", "true", '"annotation"'])
def test_given_invalid_annotation_when_loaded_then_blob_is_identified(payload: str, mocker: MockerFixture) -> None:
    """Require each annotation to contain a complete JSON object."""
    client = mocker.Mock(spec=ContainerClient)
    client.list_blobs.return_value = [BlobProperties(name="gold/document.json")]
    client.download_blob.return_value.readall.return_value = payload

    with pytest.raises(ValueError, match=r"Blob 'gold/document\.json'.*JSON object"):
        AzureBlobGoldenDatasetStore(client).load("gold")


def test_given_invalid_utf8_when_loaded_then_blob_is_identified(mocker: MockerFixture) -> None:
    """Report the offending blob while retaining the original decoding failure."""
    client = mocker.Mock(spec=ContainerClient)
    client.list_blobs.return_value = [BlobProperties(name="gold/document.json")]
    error = UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    client.download_blob.return_value.readall.side_effect = error

    with pytest.raises(ValueError, match=r"Blob 'gold/document\.json'.*UTF-8") as caught:
        AzureBlobGoldenDatasetStore(client).load("gold")

    assert caught.value.__cause__ is error


@pytest.mark.parametrize("blob_name", ["gold/.json", "gold/ .json"])
def test_given_empty_document_id_when_loaded_then_no_download(blob_name: str, mocker: MockerFixture) -> None:
    """Reject filenames that cannot identify a document."""
    client = mocker.Mock(spec=ContainerClient)
    client.list_blobs.return_value = [BlobProperties(name=blob_name)]

    with pytest.raises(ValueError, match="nonempty document ID"):
        AzureBlobGoldenDatasetStore(client).load("gold")

    client.download_blob.assert_not_called()


def test_given_colliding_document_ids_when_loaded_then_annotations_are_not_overwritten(mocker: MockerFixture) -> None:
    """Reject case-insensitive extension collisions rather than dropping gold."""
    client = mocker.Mock(spec=ContainerClient)
    client.list_blobs.return_value = [BlobProperties(name="gold/document.json"), BlobProperties(name="gold/document.JSON")]
    client.download_blob.return_value.readall.return_value = "{}"

    with pytest.raises(ValueError, match="Duplicate document ID 'document'"):
        AzureBlobGoldenDatasetStore(client).load("gold")

    client.download_blob.assert_called_once_with("gold/document.json", encoding="utf-8-sig")


@pytest.mark.parametrize("blob_name", [None, 42, "gold-other/document.json"])
def test_given_invalid_provider_blob_name_when_loaded_then_no_download(blob_name: object, mocker: MockerFixture) -> None:
    """Enforce the provider name and prefix contract before downloading."""
    client = mocker.Mock(spec=ContainerClient)
    client.list_blobs.return_value = [BlobProperties(name=blob_name)]

    with pytest.raises(RuntimeError, match="Azure returned invalid blob name"):
        AzureBlobGoldenDatasetStore(client).load("gold")

    client.download_blob.assert_not_called()


@pytest.mark.parametrize("failure_point", ["listing", "pagination", "download", "read"])
def test_given_provider_failure_when_loaded_then_original_error_propagates(failure_point: str, mocker: MockerFixture) -> None:
    """Return no partial data and never close the caller's client on failure."""
    client = mocker.MagicMock(spec=ContainerClient)
    error = HttpResponseError("synthetic storage failure")
    client.list_blobs.return_value = [BlobProperties(name="gold/document.json")]
    client.download_blob.return_value.readall.return_value = "{}"

    def failing_pages() -> Iterator[BlobProperties]:
        """Yield one valid blob before a later listing page fails."""
        yield BlobProperties(name="gold/document.json")
        raise error

    if failure_point == "listing":
        client.list_blobs.side_effect = error
    elif failure_point == "pagination":
        client.list_blobs.return_value = failing_pages()
    elif failure_point == "download":
        client.download_blob.side_effect = error
    else:
        client.download_blob.return_value.readall.side_effect = error

    with pytest.raises(HttpResponseError) as caught:
        AzureBlobGoldenDatasetStore(client).load("gold")

    assert caught.value is error
    client.close.assert_not_called()
    client.__enter__.assert_not_called()


def test_given_loaded_gold_when_passed_to_offline_pipeline_then_independent_references_are_used(raw: JsonObject, quantity_schema: type[EvaluationModel], mocker: MockerFixture) -> None:
    """Keep persistence separate while producing directly usable offline gold."""
    client = mocker.Mock(spec=ContainerClient)
    gold = {"items": [{"quantity": 12000}, {"quantity": 8000}]}
    client.list_blobs.return_value = [BlobProperties(name="gold/document.json")]
    client.download_blob.return_value.readall.return_value = json.dumps(gold)
    dataset = AzureBlobGoldenDatasetStore(client).load("gold")
    pipeline = OfflineEvaluationPipeline(OfflineEvaluationConfig(extraction_model=quantity_schema), runner=LocalRunner())

    report = pipeline.run([{"id": document_id, "cu_result": raw, "ground_truth": annotation} for document_id, annotation in dataset.items()])

    assert report["documents"] == 1
    assert report["metrics"]["field"]["extraction.pooled_field_f1"] == 1.0
    assert report["metrics"]["structural"]["extraction.table_row.tp"] == 2
