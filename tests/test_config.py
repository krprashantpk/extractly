"""Validate explicit policy before any provider operation."""

import pytest

from extractly import EvaluationConfig, FieldRule, ModelSettings, OfflineEvaluationConfig, OnlineEvaluationConfig
from extractly.types import JsonObject


@pytest.mark.parametrize("settings", [
    {"groundednes": True},
    {"groundedness": True},
    {"max_context_characters": True},
    {"max_fields_per_document": 0},
    {"groundedness_threshold": 6},
    {"thresholds": {"extraction.micro_f1": float("nan")}},
    {"fields": {"quantity": {}}},
    {"fields": {"/bad~3name": {}}},
    {"fields": {"/segments//quantity": {}}},
    {"tables": {"/items": {"keys": [""]}}},
    {"tables": {"/items": {"keys": ["id", "id"]}}},
    {"extraction_schema": {"$ref": "https://example.invalid/schema.json"}},
    {"extraction_schema": {"$dynamicRef": "https://example.invalid/schema.json"}},
    {"extraction_schema": {"$recursiveRef": "https://example.invalid/schema.json"}},
])
def test_given_invalid_policy_when_loaded_then_rejected(settings: JsonObject) -> None:
    """Reject typos, invalid pointers, remote schemas, and silent coercion."""
    with pytest.raises(ValueError):
        EvaluationConfig.model_validate(settings)


@pytest.mark.parametrize("settings", [
    {"groundedness": True},
    {"field_labels": True},
    {"completeness": True},
    {"groundedness_threshold": 6},
    {"max_fields_per_document": True},
    {"max_fields_per_document": 0},
    {"text_f1": True},
])
def test_given_invalid_online_policy_when_loaded_then_rejected(settings: JsonObject) -> None:
    """Keep online semantic prerequisites independent from offline diagnostics."""
    with pytest.raises(ValueError):
        OnlineEvaluationConfig.model_validate(settings)


@pytest.mark.parametrize("settings", [
    {"model": {"azure_endpoint": "https://example.invalid", "azure_deployment": "judge"}},
    {"groundedness": False},
    {"field_labels": False},
    {"completeness": False},
    {"max_fields_per_document": 2000},
])
def test_given_online_settings_when_offline_loaded_then_rejected(settings: JsonObject) -> None:
    """Do not silently ignore settings intended for the other workflow."""
    with pytest.raises(ValueError):
        OfflineEvaluationConfig.model_validate(settings)


def test_given_separate_policies_when_loaded_then_only_common_contract_shared() -> None:
    """Keep each workflow's settings out of the other's public schema."""
    assert "model" not in OfflineEvaluationConfig.model_fields
    assert "text_f1" not in OnlineEvaluationConfig.model_fields
    assert "text_f1" in OfflineEvaluationConfig.model_fields
    assert "model" in OnlineEvaluationConfig.model_fields


@pytest.mark.parametrize("settings", [
    {"absolute_tolerance": "-1"},
    {"absolute_tolerance": "NaN"},
    {"relative_tolerance": "not-decimal"},
    {"unit_field": "unit"},
    {"unit_scales": {"kg": "1"}},
    {"unit_field": "unit", "unit_scales": {"kg": "0"}},
])
def test_given_invalid_decimal_policy_when_loaded_then_rejected(settings: JsonObject) -> None:
    """Require finite tolerances and explicit positive unit factors."""
    with pytest.raises(ValueError):
        FieldRule.model_validate(settings)


@pytest.mark.parametrize("endpoint", [
    "http://example.invalid", "https://user:password@example.invalid",
    "https://example.invalid?key=value", "https://example.invalid#fragment",
])
def test_given_unsafe_endpoint_when_loaded_then_rejected(endpoint: str) -> None:
    """Disallow accidental credentials or key-bearing queries in model endpoints."""
    with pytest.raises(ValueError):
        ModelSettings(azure_endpoint=endpoint, azure_deployment="judge")


def test_given_explicit_key_environment_when_resolved_then_key_not_serialized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolve a caller-chosen environment variable without copying it into policy."""
    monkeypatch.setenv("EXTRACTLY_TEST_KEY", "synthetic-key")
    model = ModelSettings(azure_endpoint="https://example.invalid", azure_deployment="judge", api_key_env="EXTRACTLY_TEST_KEY")

    configuration = model.configuration()

    assert configuration["api_key"] == "synthetic-key"
    assert "synthetic-key" not in model.model_dump_json()


def test_given_missing_key_environment_when_resolved_then_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Do not silently substitute default authentication."""
    monkeypatch.delenv("EXTRACTLY_TEST_KEY", raising=False)
    model = ModelSettings(azure_endpoint="https://example.invalid", azure_deployment="judge", api_key_env="EXTRACTLY_TEST_KEY")

    with pytest.raises(ValueError, match="absent or empty"):
        model.configuration()
