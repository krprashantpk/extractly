"""Exercise schema-derived deterministic table scoring without field policy or inference."""

from copy import deepcopy
from typing import cast

import pytest
from pydantic import BaseModel, Field

from extractly.comparison import EvaluationModel, field_comparator
from extractly.evaluators.field import FieldEvaluator
from extractly.evaluators.structural import StructuralEvaluator
from extractly.types import JsonObject


class Row(BaseModel):
    """Describe a structured row independently of field-value validation."""

    quantity: int


class Extraction(BaseModel):
    """Select an aliased table and an optional independent table."""

    items: list[Row] = Field(alias="table/rows~")
    other: list[Row] | None = None


@pytest.mark.parametrize(("values", "expected"), [([1, 2], [2, 0, 0, 1.0]), ([1], [1, 0, 1, 0.667]), ([1, 2, 3], [2, 1, 0, 0.8]), ([], [0, 0, 2, 0.0])])
def test_given_row_counts_when_called_then_positional_occurrences_drive_metrics(values: list[int], expected: list[int | float]) -> None:
    """Measure row presence separately from scalar values and optional missing tables."""
    candidate = {"table/rows~": [{"quantity": value} for value in values]}
    gold = {"table/rows~": [{"quantity": 1}, {"quantity": 2}]}

    result = StructuralEvaluator()(candidate, gold, Extraction)

    assert [result["metrics"][f"extraction.table_row.{name}"] for name in ("tp", "fp", "fn", "f1")] == expected
    assert result["metrics"]["extraction.table_count.tp"] == 1
    assert next(table for table in result["tables"] if table["path"] == "/table~1rows~0")["extracted_row_count"] == len(values)


@pytest.mark.parametrize("candidate", [{}, {"table/rows~": None}, {"table/rows~": "invalid"}])
def test_given_absent_table_when_called_then_missing_rows_and_table_are_counted(candidate: JsonObject) -> None:
    """Distinguish a missing or invalid container from an explicitly empty table."""
    result = StructuralEvaluator()(candidate, {"table/rows~": [{"quantity": 1}]}, Extraction)

    assert result["metrics"]["extraction.table_count.fn"] == 1
    assert result["metrics"]["extraction.table_row.fn"] == 1


@pytest.mark.parametrize(("candidate", "gold", "expected"), [({"table/rows~": []}, {"table/rows~": []}, [1, 0, 0]), ({"other": []}, {}, [0, 1, 0]), ({}, {"other": []}, [0, 0, 1]), ({}, {}, [0, 0, 0])])
def test_given_empty_or_missing_tables_when_called_then_presence_is_not_row_count(candidate: JsonObject, gold: JsonObject, expected: list[int]) -> None:
    """Count each observed empty list as one table without filling required defaults."""
    result = StructuralEvaluator()(candidate, gold, Extraction)

    assert [result["metrics"][f"extraction.table_count.{name}"] for name in ("tp", "fp", "fn")] == expected
    assert result["metrics"]["extraction.table_row.f1"] is None


def test_given_invalid_and_reordered_values_when_called_then_only_occurrence_counts_are_scored() -> None:
    """Do not coerce, validate, or approve row identities through count equality."""
    candidate = {"table/rows~": [{"quantity": "invalid"}, {"quantity": 1}], "ignored": [1, 2]}
    original = deepcopy(candidate)
    gold = {"table/rows~": [{"quantity": 1}, {"quantity": 2}]}

    result = StructuralEvaluator()(candidate, gold, Extraction)

    assert result["metrics"]["extraction.table_row.f1"] == 1.0
    assert set(result) == {"metrics", "tables"}
    assert all("table_row" in name or "table_count" in name for name in result["metrics"])
    assert candidate == original


def test_given_missing_nested_table_when_called_then_concrete_parent_paths_are_preserved() -> None:
    """Do not pool nested rows across different parent occurrences."""
    class Group(BaseModel):
        """Contain one nested row table."""

        rows: list[Row]

    class Groups(BaseModel):
        """Select both group records and their nested row lists."""

        groups: list[Group]

    candidate = {"groups": [{"rows": [{"quantity": 1}, {"quantity": 2}]}]}
    gold = {"groups": [{"rows": [{"quantity": 1}]}, {"rows": [{"quantity": 2}]}]}

    result = StructuralEvaluator()(candidate, gold, Groups)

    assert [result["metrics"][f"extraction.table_count.{name}"] for name in ("tp", "fp", "fn")] == [2, 0, 1]
    assert [result["metrics"][f"extraction.table_row.{name}"] for name in ("tp", "fp", "fn")] == [2, 1, 2]
    assert {table["path"] for table in result["tables"]} == {"/groups", "/groups/0/rows", "/groups/1/rows"}


def test_given_whole_list_comparator_when_scored_then_structural_counts_remain_independent() -> None:
    """Keep a tolerant field match from hiding duplicate or extra row occurrences."""
    class ComparedRow(EvaluationModel):
        """Declare a typed quantity for the whole-list comparator."""

        quantity: int

    class ComparedExtraction(EvaluationModel):
        """Accept set-equivalent field values without owning structural policy."""

        items: list[ComparedRow]

        @staticmethod
        @field_comparator("items")
        def same_values(expected: list[ComparedRow], actual: list[ComparedRow]) -> bool:
            """Ignore multiplicity only for this configured field comparison."""
            return {row.quantity for row in expected} == {row.quantity for row in actual}

    candidate = {"items": [{"quantity": 2}, {"quantity": 1}, {"quantity": 1}]}
    gold = {"items": [{"quantity": 1}, {"quantity": 2}]}

    fields = FieldEvaluator()(candidate, gold, ComparedExtraction)
    structure = StructuralEvaluator()(candidate, gold, ComparedExtraction)

    assert fields["exact_match"] == 1
    assert fields["field_counts"] == {"/items": [1, 0, 0]}
    assert structure["metrics"]["extraction.table_row.fp"] == 1


def test_given_scalar_array_when_called_then_it_is_not_invented_as_a_record_table() -> None:
    """Require structured record arrays instead of treating every list as a table."""
    class Tags(BaseModel):
        """Declare an array that has no row-record structure."""

        tags: list[str]

    with pytest.raises(ValueError, match="structured object records"):
        StructuralEvaluator()({"tags": ["a"]}, {"tags": ["a"]}, Tags)


def test_given_invalid_model_when_called_then_input_contract_is_rejected() -> None:
    """Validate the model boundary before attempting schema discovery."""
    with pytest.raises(TypeError, match="BaseModel"):
        StructuralEvaluator()({}, {}, cast(type[BaseModel], dict))