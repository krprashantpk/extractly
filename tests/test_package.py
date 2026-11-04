"""Verify the installable public surface is a Python library, not a CLI."""

from importlib.metadata import distribution
from importlib.util import find_spec
from inspect import signature
from pkgutil import iter_modules

import extractly
import extractly.cu as cu
import extractly.evaluators as evaluators
import extractly.store as store
from extractly import (
    AzureBlobGoldenDatasetStore,
    FieldEvaluator,
    GoldenDatasetStore,
    LLMFieldEvaluator,
    LLMStructuralEvaluator,
    OfflineEvaluationPipeline,
    OnlineEvaluationPipeline,
    ParsedCUResult,
    ParsedCUSegment,
    SchemaEvaluator,
    StructuralEvaluator,
    parse_cu_result,
)
from extractly.evaluators.field import FieldEvaluator as PackagedFieldEvaluator
from extractly.evaluators.schema import SchemaEvaluator as PackagedSchemaEvaluator
from extractly.store.base import GoldenDatasetStore as PackagedGoldenDatasetStore
from extractly.store.blob import AzureBlobGoldenDatasetStore as PackagedAzureBlobGoldenDatasetStore


def test_given_installed_package_when_inspected_then_no_cli_entry_points() -> None:
    """Do not install an Extractly command or executable package module."""
    assert not [entry for entry in distribution("extractly").entry_points if entry.group == "console_scripts"]
    assert find_spec("extractly.cli") is None
    assert find_spec("extractly.__main__") is None
    assert find_spec("extractly.framework") is None


def test_given_public_api_when_imported_then_independent_evaluators() -> None:
    """Expose dedicated workflows rather than an implicit mode dispatcher."""
    assert "EvaluationFramework" not in extractly.__all__
    assert "OfflineEvaluationPipeline" in extractly.__all__
    assert "OnlineEvaluationPipeline" in extractly.__all__
    assert "EvaluationModel" in extractly.__all__
    assert "FieldRule" not in extractly.__all__
    assert not hasattr(extractly, "FieldRule")
    assert "mode" not in signature(OfflineEvaluationPipeline.run).parameters
    assert not hasattr(OfflineEvaluationPipeline, "evaluate")
    assert "mode" not in signature(OnlineEvaluationPipeline.run).parameters
    assert not hasattr(extractly, "OfflineEvaluator")
    assert not hasattr(extractly, "OnlineEvaluator")


def test_given_cu_subpackage_when_imported_then_public_api_is_consistent() -> None:
    """Expose one CU parser module through the root and subpackage."""
    assert parse_cu_result is cu.parse_cu_result
    assert ParsedCUResult is cu.ParsedCUResult
    assert ParsedCUSegment is cu.ParsedCUSegment
    assert "parse_cu_result" in extractly.__all__
    assert not hasattr(extractly, "CUAdapter")
    assert not hasattr(extractly, "build_ground_truth")
    assert {module.name for module in iter_modules(cu.__path__)} == {"cu_result_parser"}
    assert find_spec("extractly.adapter") is None


def test_given_evaluators_subpackage_when_imported_then_public_api_is_consistent() -> None:
    """Preserve evaluator and ratio imports while exposing dedicated modules."""
    assert FieldEvaluator is evaluators.FieldEvaluator is PackagedFieldEvaluator
    assert StructuralEvaluator is evaluators.StructuralEvaluator
    assert LLMFieldEvaluator is evaluators.LLMFieldEvaluator
    assert LLMStructuralEvaluator is evaluators.LLMStructuralEvaluator
    assert SchemaEvaluator is evaluators.SchemaEvaluator is PackagedSchemaEvaluator
    assert evaluators.ratios(1, 0, 0) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert not hasattr(evaluators, "ExtractionEvaluator")
    assert not hasattr(evaluators, "OnlineFieldEvaluator")


def test_given_store_subpackage_when_imported_then_public_api_is_consistent() -> None:
    """Expose the same golden-dataset stores through the root and subpackage."""
    assert GoldenDatasetStore is store.GoldenDatasetStore is PackagedGoldenDatasetStore
    assert AzureBlobGoldenDatasetStore is store.AzureBlobGoldenDatasetStore is PackagedAzureBlobGoldenDatasetStore
    assert {"AzureBlobGoldenDatasetStore", "GoldenDatasetStore"} <= set(extractly.__all__)
    assert set(store.__all__) == {"AzureBlobGoldenDatasetStore", "GoldenDatasetStore"}
