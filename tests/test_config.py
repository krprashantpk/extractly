"""Validate explicit policy before any provider operation."""

import pytest

from extractly import EvaluationConfig, EvaluationModel, OfflineEvaluationConfig, OnlineEvaluationConfig
from extractly.types import JsonObject


@pytest.mark.parametrize("settings", [
    {"groundednes": True},
    {"groundedness": True},
    {"max_context_characters": True},
    {"max_fields_per_document": 0},
    {"groundedness_threshold": 6},
    {"thresholds": {"extraction.pooled_field_f1": float("nan")}},
    {"fields": {"quantity": {}}},
    {"evaluation_model": None},
    {"tables": {"items": {}}},
    {"tables": {"/bad~3name": {}}},
    {"tables": {"/segments//items": {}}},
    {"include_fields": []},
    {"tables": {"/items": {"keys": [""]}}},
    {"tables": {"/items": {"keys": ["id", "id"]}}},
    {"tables": {"/items": {"keys": ["id"]}}},
    {"tables": {"/items": {"level": "row"}}},
    {"tables": {"/items": {"level": "instance"}}},
    {"extraction_schema": {"type": "object"}},
    {"extraction_model": {"type": "object"}},
])
def test_given_invalid_policy_when_loaded_then_rejected(settings: JsonObject) -> None:
    """Reject typos, invalid pointers, remote schemas, and silent coercion."""
    with pytest.raises(ValueError):
        EvaluationConfig.model_validate(settings)


@pytest.mark.parametrize("level", ["table_row", "table_count"])
def test_given_table_metric_level_when_loaded_then_positional_policy(level: str) -> None:
    """Accept only the renamed structural levels without an identity-key option."""
    config = EvaluationConfig.model_validate({"tables": {"/items": {"level": level}}})

    assert config.tables["/items"].level == level
    assert "keys" not in type(config.tables["/items"]).model_fields


@pytest.mark.parametrize("settings", [
    {"model": None},
    {"groundedness": False},
    {"groundedness": True},
    {"field_labels": False},
    {"field_labels": True},
    {"completeness": False},
    {"completeness": True},
    {"groundedness_threshold": 3},
    {"groundedness_threshold": 6},
    {"field_label_instructions": ""},
    {"completeness_instructions": ""},
    {"max_fields_per_document": True},
    {"max_fields_per_document": 0},
    {"text_f1": True},
])
def test_given_invalid_online_policy_when_loaded_then_rejected(settings: JsonObject) -> None:
    """Reject removed graders and invalid Decision budgets rather than ignoring them."""
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
    assert "model" not in OnlineEvaluationConfig.model_fields
    assert "max_fields_per_document" in OnlineEvaluationConfig.model_fields


@pytest.mark.parametrize(("path", "selected"), [("/segments/0/fields/items/0/quantity", True), ("/segments/0/fields/items/1/quantity", True), ("/segments/0/fields/items/0/name", False), ("/other/quantity", False)])
def test_given_field_selection_when_matching_then_wildcards_cover_occurrences(path: str, selected: bool) -> None:
    """Match selected occurrences rather than similarly named unrelated fields."""
    class Quantity(EvaluationModel):
        """Declare the selected item property."""

        quantity: int

    class Items(EvaluationModel):
        """Apply item selection to every array occurrence."""

        items: list[Quantity]

    config = EvaluationConfig(extraction_model=Items)

    result = config.selects_path(path)

    assert result is selected


def test_given_extraction_model_when_configured_then_compiled_schema_exposed() -> None:
    """Distinguish the supplied model from its private compiled evaluation policy."""
    class Quantity(EvaluationModel):
        """Declare the selected quantity."""

        quantity: int

    config = EvaluationConfig(extraction_model=Quantity)

    schema = config.extraction_schema

    assert config.extraction_model is Quantity
    assert schema is not None
    assert schema.model is Quantity
    assert schema.schema == Quantity.model_json_schema(mode="validation")
    assert config.model_dump()["extraction_model"] is Quantity
    assert config.model_dump(mode="json")["extraction_model"] == schema.describe()
    assert "extraction_schema" not in config.model_dump()
    assert "_extraction_schema" not in config.model_dump()


def test_given_changed_extraction_model_when_assigned_then_schema_recompiled() -> None:
    """Rebuild or clear compiled selection whenever the supplied model changes."""
    class Quantity(EvaluationModel):
        """Declare a selected quantity."""

        quantity: int

    config = EvaluationConfig()
    assert config.extraction_schema is None

    config.extraction_model = Quantity

    assert config.extraction_schema is not None
    assert config.extraction_schema.model is Quantity
    assert not config.selects_path("/segments/0/fields/unrelated")

    config.extraction_model = None

    assert config.extraction_schema is None
    assert config.selects_path("/segments/0/fields/unrelated")
    assert config.model_dump(mode="json")["extraction_model"] is None
