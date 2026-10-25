"""Validate explicit policy before any provider operation."""

import pytest

from extractly import EvaluationConfig, EvaluationModel, ModelSettings, OfflineEvaluationConfig, OnlineEvaluationConfig
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
    {"evaluation_model": None},
    {"tables": {"items": {}}},
    {"tables": {"/bad~3name": {}}},
    {"tables": {"/segments//items": {}}},
    {"include_fields": []},
    {"include_fields": ["quantity"]},
    {"include_fields": ["/bad~3name"]},
    {"include_fields": ["/items//quantity"]},
    {"include_fields": ["/quantity", "/quantity"]},
    {"include_fields": "/quantity"},
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


def test_given_raw_json_schema_when_loaded_then_model_class_required() -> None:
    """Reject a second schema language instead of maintaining competing policies."""
    with pytest.raises(ValueError):
        EvaluationConfig.model_validate({"extraction_schema": {"type": "object"}})


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


@pytest.mark.parametrize(("path", "selected"), [("/segments/0/fields/items/0/quantity", True), ("/segments/0/fields/items/1/quantity", True), ("/segments/0/fields/items/0/name", False), ("/other/quantity", False)])
def test_given_field_selection_when_matching_then_wildcards_cover_occurrences(path: str, selected: bool) -> None:
    """Match selected occurrences rather than similarly named unrelated fields."""
    class Quantity(EvaluationModel):
        """Declare the selected item property."""

        quantity: int

    class Items(EvaluationModel):
        """Apply item selection to every array occurrence."""

        items: list[Quantity]

    config = EvaluationConfig(extraction_schema=Items)

    result = config.selects_path(path)

    assert result is selected
