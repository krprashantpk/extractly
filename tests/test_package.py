"""Verify the installable public surface is a Python library, not a CLI."""

from importlib.metadata import distribution
from importlib.util import find_spec
from inspect import signature

import extractly
import extractly.evaluators as evaluators
from extractly import (
    CUAdapter,
    ExtractionEvaluator,
    OfflineEvaluator,
    OnlineEvaluator,
    SchemaEvaluator,
    build_ground_truth,
)
from extractly.cu import CUAdapter as PackagedCUAdapter
from extractly.cu.ground_truth import build_ground_truth as packaged_build_ground_truth
from extractly.evaluators.extraction import ExtractionEvaluator as PackagedExtractionEvaluator
from extractly.evaluators.schema import SchemaEvaluator as PackagedSchemaEvaluator


def test_given_installed_package_when_inspected_then_no_cli_entry_points() -> None:
    """Do not install an Extractly command or executable package module."""
    assert not [entry for entry in distribution("extractly").entry_points if entry.group == "console_scripts"]
    assert find_spec("extractly.cli") is None
    assert find_spec("extractly.__main__") is None
    assert find_spec("extractly.framework") is None


def test_given_public_api_when_imported_then_independent_evaluators() -> None:
    """Expose dedicated workflows rather than an implicit mode dispatcher."""
    assert "EvaluationFramework" not in extractly.__all__
    assert "OfflineEvaluator" in extractly.__all__
    assert "OnlineEvaluator" in extractly.__all__
    assert "EvaluationModel" in extractly.__all__
    assert "FieldRule" not in extractly.__all__
    assert not hasattr(extractly, "FieldRule")
    assert "mode" not in signature(OfflineEvaluator.run).parameters
    assert "mode" not in signature(OnlineEvaluator.run).parameters


def test_given_cu_subpackage_when_imported_then_public_api_is_consistent() -> None:
    """Expose CU helpers through the root and subpackage without a legacy shim."""
    assert CUAdapter is PackagedCUAdapter
    assert build_ground_truth is packaged_build_ground_truth
    assert "build_ground_truth" in extractly.__all__
    assert find_spec("extractly.adapter") is None


def test_given_evaluators_subpackage_when_imported_then_public_api_is_consistent() -> None:
    """Preserve evaluator and ratio imports while exposing dedicated modules."""
    assert ExtractionEvaluator is evaluators.ExtractionEvaluator is PackagedExtractionEvaluator
    assert SchemaEvaluator is evaluators.SchemaEvaluator is PackagedSchemaEvaluator
    assert evaluators.ratios(1, 0, 0) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
