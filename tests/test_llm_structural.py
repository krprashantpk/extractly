"""Exercise structural counts and identity review without credentials or network calls."""

import json
from copy import deepcopy

import httpx
import pytest
from openai import OpenAI
from pydantic import BaseModel, Field

from extractly.evaluators import LLMStructuralEvaluator
from extractly.evaluators.llm_structural import (
    STRUCTURAL_CHECKS,
    STRUCTURAL_PROMPT,
    StructuralEvaluation,
    _score_structure,
    _table_targets,
)
from extractly.types import JsonObject


class Row(BaseModel):
    """Declare record structure independently of scalar value correctness."""

    identity: str
    quantity: float


class Extraction(BaseModel):
    """Select an aliased table and an optional second table."""

    rows: list[Row] = Field(alias="table/rows")
    other: list[Row] | None = None


def judgment(*, rows: int | None = 2, tables: int | None = 1, failed_check: str | None = None, ambiguous_check: str | None = None) -> StructuralEvaluation:
    """Construct controlled source-count and structural-check outputs."""
    return StructuralEvaluation.model_validate({"tables": [{"path": "/items", "source_table_count": tables, "source_row_count": rows, "source_excerpt": None, "explanation": "Mock source inventory"}], "checks": [{"name": name, "verdict": "fail" if name == failed_check else "ambiguous" if name == ambiguous_check else "pass", "paths": [], "source_excerpt": None, "explanation": "Mock structural judgment"} for name in STRUCTURAL_CHECKS], "summary": "Mock structural judgment"})


@pytest.mark.parametrize(("extracted", "source", "expected"), [(2, 2, [2, 0, 0, 1.0]), (1, 2, [1, 0, 1, 0.667]), (3, 2, [2, 1, 0, 0.8]), (0, 0, [0, 0, 0, None])])
def test_given_source_counts_when_scored_then_occurrence_counts_follow_offline_formulas(extracted: int, source: int, expected: list[int | float | None]) -> None:
    """Measure count agreement independently of scalar field correctness."""
    targets = [{"path": "/items", "extracted_table_count": 1, "extracted_row_count": extracted}]

    result = _score_structure(judgment(rows=source), targets)

    assert [result["metrics"][f"extraction.table_row.{name}"] for name in ("tp", "fp", "fn", "f1")] == expected
    assert result["metrics"]["extraction.table_count.tp"] == 1


@pytest.mark.parametrize("name", STRUCTURAL_CHECKS)
def test_given_equal_counts_with_integrity_failure_when_scored_then_review_required(name: str) -> None:
    """Do not let equal counts approve duplicates, missing rows, merges, splits, or reordering."""
    targets = [{"path": "/items", "extracted_table_count": 1, "extracted_row_count": 2}]

    result = _score_structure(judgment(failed_check=name), targets)

    assert result["metrics"]["extraction.table_row.f1"] == 1.0
    assert result["review_required"] is True


@pytest.mark.parametrize("name", STRUCTURAL_CHECKS)
def test_given_ambiguous_integrity_check_when_scored_then_review_required(name: str) -> None:
    """Retain unresolved structural judgments without a confidence threshold."""
    result = _score_structure(judgment(ambiguous_check=name), [{"path": "/items", "extracted_table_count": 1, "extracted_row_count": 2}])

    assert result["review_required"] is True


def test_given_unknown_source_rows_when_scored_then_row_metrics_unknown_but_table_metrics_retained() -> None:
    """Never turn an unknown source count into zero or a success-shaped row score."""
    result = _score_structure(judgment(rows=None), [{"path": "/items", "extracted_table_count": 1, "extracted_row_count": 2}])

    assert result["metrics"]["extraction.table_row.tp"] is None
    assert result["metrics"]["extraction.table_row.f1"] is None
    assert result["metrics"]["extraction.table_count.tp"] == 1
    assert result["review_required"] is True


@pytest.mark.parametrize("source_tables", [0, 1, 2])
def test_given_source_table_count_when_scored_then_table_metrics_follow_container_counts(source_tables: int) -> None:
    """Count logical source tables rather than physical page fragments."""
    result = _score_structure(judgment(tables=source_tables), [{"path": "/items", "extracted_table_count": 1, "extracted_row_count": 2}])

    assert [result["metrics"][f"extraction.table_count.{name}"] for name in ("tp", "fp", "fn")] == [min(1, source_tables), max(1 - source_tables, 0), max(source_tables - 1, 0)]


def test_given_empty_and_missing_arrays_when_targets_prepared_then_only_empty_list_is_table() -> None:
    """Preserve aliases and distinguish an empty table from an absent optional array."""
    targets = _table_targets({"table/rows": []}, Extraction.model_json_schema())

    assert targets == [{"path": "/table~1rows", "extracted_table_count": 1, "extracted_row_count": 0, "rows": []}, {"path": "/other", "extracted_table_count": 0, "extracted_row_count": 0, "rows": []}]


def test_given_four_inputs_when_called_then_payload_and_response_preserve_structure() -> None:
    """Use a real SDK parser over a local transport without owning caller cleanup."""
    class Selected(BaseModel):
        """Select only one row array while retaining other candidate data as context."""

        items: list[Row]

    candidate = {"items": [{"identity": "B", "quantity": 0}, {"identity": "A", "quantity": 5}], "unselected": []}
    original = deepcopy(candidate)
    analyzer = {"analyzerId": "test", "fieldSchema": {"fields": {"items": {"type": "array"}}}}
    requests: list[JsonObject] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """Record the request and return a row-order failure with equal row counts."""
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "local", "object": "chat.completion", "created": 0, "model": "test", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": judgment(failed_check="row_order").model_dump_json()}}]})

    with OpenAI(api_key="local-test", base_url="https://example.test/v1/", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        result = LLMStructuralEvaluator(client, "deployment")("A then B", candidate, analyzer, Selected)
        payload = json.loads(requests[0]["messages"][1]["content"])

        assert payload["extracted_values"] == candidate == original
        assert payload["analyzer_definition"] == analyzer
        assert payload["extraction_schema"] == Selected.model_json_schema()
        assert payload["table_targets"] == [{"path": "/items", "extracted_table_count": 1, "extracted_row_count": 2, "rows": [{"path": "/items/0", "value": candidate["items"][0]}, {"path": "/items/1", "value": candidate["items"][1]}]}]
        assert requests[0]["messages"][0]["content"] == STRUCTURAL_PROMPT
        assert set(result) == {"judgment", "metrics", "review_required", "deployment", "latency_seconds", "usage"}
        assert result["metrics"]["extraction.table_row.f1"] == 1.0
        assert result["review_required"] is True
        assert "confidence" not in requests[0]["response_format"]["json_schema"]["schema"]["$defs"]["StructuralCheck"]["properties"]
        assert "needs_document_context" not in json.dumps(requests[0]["response_format"])
        assert not client.is_closed()


def test_given_negative_source_count_when_scored_then_provider_contract_failure_is_visible() -> None:
    """Reject impossible source counts rather than fabricating count metrics."""
    with pytest.raises(RuntimeError, match="nonnegative"):
        _score_structure(judgment(rows=-1), [{"path": "/items", "extracted_table_count": 1, "extracted_row_count": 2}])