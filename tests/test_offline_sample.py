"""Verify offline sample loading without depending on private source documents."""

import importlib
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def offline_sample(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Provide the notebook's loader with an isolated, source-paired annotation."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("experiment.offline_evaluation")
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "source.txt").write_text("Quantity: 1", encoding="utf-8")
    (samples / "source.txt.json").write_text("{}", encoding="utf-8")
    annotations = {
        "format": "extractly-source-annotations-v1",
        "line_item_defaults": {},
        "documents": [{
            "id": "source",
            "source_file": "source.txt",
            "cu_result_file": "source.txt.json",
            "source_row_count": 1,
            "source_total_quantity": "1",
            "row_identity": None,
            "line_item_defaults": {},
            "ground_truth": {"segments": [{"fields": {"lineItems": [{"quantity": 1}]}}]},
        }],
    }
    annotations_path = tmp_path / "golden_dataset.json"
    annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
    monkeypatch.setattr(module, "EXPERIMENT_DIRECTORY", tmp_path)
    monkeypatch.setattr(module, "ANNOTATIONS_PATH", annotations_path)
    return module


def test_given_standalone_fixture_when_loaded_then_only_annotated_sources_returned(offline_sample: ModuleType) -> None:
    """Exclude unrelated JSON fixtures from the source-backed benchmark."""
    fixture_path = offline_sample.EXPERIMENT_DIRECTORY / "samples" / "synthetic_two_page_weighbridge.json"
    fixture_path.write_text("not a prediction", encoding="utf-8")

    annotations, documents = offline_sample.load_documents()

    assert documents == [{"id": "source", "cu_result": {}, "ground_truth": annotations["documents"][0]["ground_truth"]}]


@pytest.mark.parametrize("inventory", ["missing_prediction", "unannotated_source", "duplicate_annotation"])
def test_given_invalid_inventory_when_loaded_then_rejected(offline_sample: ModuleType, inventory: str) -> None:
    """Keep exact coverage and uniqueness checks for source-paired predictions."""
    samples = offline_sample.EXPERIMENT_DIRECTORY / "samples"
    if inventory == "missing_prediction":
        (samples / "source.txt.json").unlink()
    elif inventory == "unannotated_source":
        (samples / "other.txt").write_text("Quantity: 2", encoding="utf-8")
        (samples / "other.txt.json").write_text("{}", encoding="utf-8")
    else:
        annotations = json.loads(offline_sample.ANNOTATIONS_PATH.read_bytes())
        duplicate = deepcopy(annotations["documents"][0])
        duplicate["id"] = "duplicate"
        annotations["documents"].append(duplicate)
        offline_sample.ANNOTATIONS_PATH.write_text(json.dumps(annotations), encoding="utf-8")

    with pytest.raises(ValueError, match="every source-paired CU result exactly once"):
        offline_sample.load_documents()