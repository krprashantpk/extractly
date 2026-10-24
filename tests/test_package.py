"""Verify the installable public surface is a Python library, not a CLI."""

from importlib.metadata import distribution
from importlib.util import find_spec
from inspect import signature

import extractly
from extractly import OfflineEvaluator, OnlineEvaluator


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
    assert "mode" not in signature(OfflineEvaluator.run).parameters
    assert "mode" not in signature(OnlineEvaluator.run).parameters
