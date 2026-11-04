"""Score model-derived table and row presence against normalized reference fields."""

from __future__ import annotations

import logging

from pydantic import BaseModel

from ..types import JsonObject
from ._shared import ratios
from .llm_structural import _table_targets

logger = logging.getLogger(__name__)


class StructuralEvaluator:
    """Compare table and row occurrence counts without evaluating scalar values.

    The model selects arrays of object records, independently of any registered
    field comparators. Each selected list is one table and its elements are rows.
    Empty lists are tables; missing, null, and non-list values are not. Nested
    tables retain concrete parent indices rather than being pooled or aligned.
    Table discovery is shared with LLMStructuralEvaluator, but no inference or
    source-document judgment is performed here.

    Metrics measure presence at corresponding table paths and positional row
    slots, not row identity, order, or scalar value correctness. Counts are
    summed across table scopes before calculating ratios. Ratios are rounded
    to three decimal places and are None when their denominator is zero.

    Per-document table-row metrics:
        extraction.table_row.tp: Positional row slots present in both reference and extraction at corresponding table paths.
        extraction.table_row.fp: Extra extracted row occurrences at their original table paths.
        extraction.table_row.fn: Missing expected row occurrences at their original table paths.
        extraction.table_row.precision: table_row_tp / (table_row_tp + table_row_fp).
        extraction.table_row.recall: table_row_tp / (table_row_tp + table_row_fn).
        extraction.table_row.f1: 2 * table_row_tp / (2 * table_row_tp + table_row_fp + table_row_fn).

    Per-document table-count metrics:
        extraction.table_count.tp: Expected and extracted list containers present at corresponding model-selected table paths, including empty lists.
        extraction.table_count.fp: Extra extracted list containers at model-selected table paths.
        extraction.table_count.fn: Missing expected list containers at model-selected table paths.
        extraction.table_count.precision: table_count_tp / (table_count_tp + table_count_fp).
        extraction.table_count.recall: table_count_tp / (table_count_tp + table_count_fn).
        extraction.table_count.f1: 2 * table_count_tp / (2 * table_count_tp + table_count_fp + table_count_fn).
    """

    def __call__(self, extracted_values: JsonObject, ground_truth: JsonObject, extraction_model: type[BaseModel]) -> JsonObject:
        """Return structural occurrence metrics and per-table count diagnostics.

        Counts compare corresponding table paths and positional row slots.
        Matching counts do not establish row identity, order, or scalar value
        correctness, so no document exact-match or approval signal is returned.
        Neither input is coerced, default-filled, or otherwise modified.

        Args:
            extracted_values: Normalized candidate fields retaining original nesting and array order.
            ground_truth: Reference fields providing observed table containers and row occurrences.
            extraction_model: Pydantic model class selecting arrays of structured object records.

        Returns:
            Metrics with extraction.table_row and extraction.table_count TP/FP/FN and precision/recall/F1, plus tables containing escaped paths and observed reference and candidate counts.

        Raises:
            TypeError: Inputs are not dictionaries or the model is not a Pydantic model class.
            ValueError: No record-array tables are selected, or model containers are recursive or ambiguous.

        Scalar validation and configured field comparisons belong to the separate
        field evaluator. No configuration, table rules, clients, or call state
        are required or retained. Undefined ratios are None; other ratios are
        rounded to three decimal places.
        """
        if not isinstance(extracted_values, dict) or not isinstance(ground_truth, dict):
            raise TypeError("extracted_values and ground_truth must be dictionaries")
        if not isinstance(extraction_model, type) or not issubclass(extraction_model, BaseModel):
            raise TypeError("extraction_model must be a Pydantic BaseModel class")
        schema = extraction_model.model_json_schema(mode="validation")
        expected_tables = {target["path"]: target for target in _table_targets(ground_truth, schema)}
        extracted_tables = {target["path"]: target for target in _table_targets(extracted_values, schema)}
        paths = sorted(expected_tables.keys() | extracted_tables.keys())
        if not paths:
            raise ValueError("extraction_model must select at least one array of structured object records")
        logger.info("Starting structural evaluation with model '%s' for %s table scopes", extraction_model.__name__, len(paths))
        totals = {"table_row": [0, 0, 0], "table_count": [0, 0, 0]}
        tables: list[JsonObject] = []
        for path in paths:
            expected = expected_tables.get(path, {})
            extracted = extracted_tables.get(path, {})
            table: JsonObject = {"path": path}
            for level, count_name in (("table_row", "row_count"), ("table_count", "table_count")):
                expected_count = expected.get(f"extracted_{count_name}", 0)
                extracted_count = extracted.get(f"extracted_{count_name}", 0)
                table[f"expected_{count_name}"] = expected_count
                table[f"extracted_{count_name}"] = extracted_count
                counts = [min(expected_count, extracted_count), max(extracted_count - expected_count, 0), max(expected_count - extracted_count, 0)]
                totals[level] = [total + count for total, count in zip(totals[level], counts, strict=True)]
            tables.append(table)
        metrics: JsonObject = {}
        for level, (tp, fp, fn) in totals.items():
            metrics.update({f"extraction.{level}.{name}": value for name, value in ({"tp": tp, "fp": fp, "fn": fn} | ratios(tp, fp, fn)).items()})
        logger.info("Completed structural evaluation with model '%s': table counts=%s row counts=%s", extraction_model.__name__, totals["table_count"], totals["table_row"])
        return {"metrics": metrics, "tables": tables}