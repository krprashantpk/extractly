"""Score extraction occurrences rather than substituting text similarity.

Any is required for recursive values in caller-defined extraction schemas.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from pydantic import ValidationError

from ..comparison import equal_values
from ..config import EvaluationConfig
from ..cu.cu_result_parser import MISSING, pointer
from ..types import JsonObject
from ._shared import ratios


class ExtractionEvaluator:
    """Compute field, table-row, and table-count metrics for one document.

    With an extraction model, only declared fields and the structure traversed
    to reach them are evaluated. Arrays traversed by the evaluator are compared
    at corresponding positional indices. A comparator registered on a whole
    list or object instead treats that container as one field occurrence; its
    internal order, multiplicity, and structure are not checked independently,
    and nested table metrics are not collected. The comparator owns those
    decisions, so the positional guarantee does not cover such containers.

    Each field comparator must return exactly one bool: True for a match and
    False for a mismatch. It receives validated values only when both sides
    are present. Explicitly allowed variation counts as a normal match in all
    field scores and exact_match; there is no separate tolerant score set.
    Without an extraction model, scalar values use default equality, with
    booleans distinguished from numbers, and containers are traversed normally.

    A wrong present field value contributes both a false positive and a false
    negative. A present expected value without a present prediction contributes
    only a false negative; a present prediction without a present expected value
    contributes only a false positive. Zero and False are present values;
    missing values, None, and empty strings do not contribute positive field
    occurrences. Two absent values add no true positive, but different absence
    representations or a missing key can still make exact_match zero. Invalid
    model gold raises; invalid present predictions are counted as mismatches.

    Counts are integers. Precision, recall, and F1 scores range from zero to
    one and are rounded to three decimal places. A score is None when its
    denominator is zero, rather than being treated as perfect agreement.

    Run-level exact-match metrics:
        extraction.exact_match: Integer 1 when every document matches completely, otherwise 0; five matching documents out of six produce 0, not 0.833.
        extraction.document_exact_match_rate: Number of completely matching documents divided by the total number of evaluated documents, rounded to three decimal places; five matching documents out of six produce 0.833.

    OfflineEvaluator computes both run-level metrics from the individual
    document results. This callable returns one document's binary exact_match,
    which appears as extraction.exact_match in that document_metrics entry.
    The run's extraction.exact_match is never the document average; that
    average belongs only to extraction.document_exact_match_rate.

    Per-document field metrics:
        tp: Number of present field-value occurrences accepted by the configured comparison policy.
        fp: Number of incorrect or extra present predicted field-value occurrences.
        fn: Number of present expected field-value occurrences missing or predicted incorrectly.
        precision: tp / (tp + fp), the fraction of predicted field-value occurrences that are correct.
        recall: tp / (tp + fn), the fraction of expected field-value occurrences captured correctly.
        f1: 2 * tp / (2 * tp + fp + fn), the harmonic balance of precision and recall.
        exact_match: Per-document integer 1 when every evaluated comparison is accepted and traversed presence and structure match; otherwise 0; explicitly allowed variation is a match, and whole-container internals are governed solely by their comparator.
        field_counts: Mapping from escaped, wildcard field paths to [tp, fp, fn] occurrence counts across this document.
        comparisons: Included only when an extraction model is configured; maps concrete field paths to match, reason, status, and prediction_path; match is boolean, reason is None or a prediction-validation explanation, and status is compared, missing, extra, or absent rather than a correctness decision.

    Per-document table-row metrics:
        table_row_tp: Number of positions containing both an expected and a predicted item in arrays configured with the table_row level.
        table_row_fp: Number of extra predicted items at those configured array paths.
        table_row_fn: Number of expected items missing at those configured array paths.
        table_row_precision: table_row_tp / (table_row_tp + table_row_fp), measuring predicted row presence rather than field correctness.
        table_row_recall: table_row_tp / (table_row_tp + table_row_fn), measuring expected row presence rather than field correctness.
        table_row_f1: 2 * table_row_tp / (2 * table_row_tp + table_row_fp + table_row_fn), balancing row presence precision and recall.

    Per-document table-count metrics:
        table_count_tp: Number of positions containing both an expected and a predicted table occurrence in arrays configured with the table_count level, such as the segments array.
        table_count_fp: Number of extra predicted table occurrences at those configured array paths.
        table_count_fn: Number of expected table occurrences missing at those configured array paths.
        table_count_precision: table_count_tp / (table_count_tp + table_count_fp), measuring predicted table presence rather than its row or field correctness.
        table_count_recall: table_count_tp / (table_count_tp + table_count_fn), measuring expected table presence rather than its row or field correctness.
        table_count_f1: 2 * table_count_tp / (2 * table_count_tp + table_count_fp + table_count_fn), balancing table presence precision and recall.

    Structural counts are collected only for configured, traversed array paths.
    Unconfigured levels retain zero counts and undefined scores. A matched row
    or table can still have incorrect values, and reordered items can retain
    matched structural positions while failing field comparison or exact match.

    Run-level field metrics:
        extraction.mean_document_precision: Average of defined per-document precision scores, giving each included document equal weight.
        extraction.mean_document_recall: Average of defined per-document recall scores, giving each included document equal weight.
        extraction.mean_document_f1: Average of defined per-document F1 scores, not F1 recomputed from the mean precision and recall.
        extraction.pooled_field_precision: Precision calculated once from TP and FP summed across all documents, giving each predicted field occurrence equal weight.
        extraction.pooled_field_recall: Recall calculated once from TP and FN summed across all documents, giving each expected field occurrence equal weight.
        extraction.pooled_field_f1: F1 calculated once from TP, FP, and FN summed across all documents, not an average of document F1 scores.

    The callable returns the unprefixed per-document keys described above.
    OfflineEvaluator creates the run-level metrics separately. Its report's
    document_metrics entries retain extraction.precision, extraction.recall,
    and extraction.f1 for each document. Other runner-aggregated scalars in
    report.metrics, including TP, FP, FN, and structural counts,
    remain document averages rather than totals.
    LocalRunner omits a mean_document score when every document's corresponding
    score is undefined; pooled_field scores remain present with None when their
    pooled denominator is zero. Empty agreement may have exact_match equal to
    1 while positive-occurrence precision, recall, and F1 remain undefined.
    """

    def __init__(self, config: EvaluationConfig) -> None:
        """Accept field comparison policies and positional table metric rules."""
        self.config = config

    def __call__(self, *, response: str, ground_truth: str, **kwargs: object) -> JsonObject:
        """Compare extraction against independent gold using the configured policy.

        Wrong nonempty values contribute both a false positive and a false
        negative. Invalid predictions count as errors; invalid gold raises.
        Without an extraction model, category and page metadata can affect
        exact match but not field F1.
        With extraction_model, only declared fields and their containing structure
        affect scores and exact match; undeclared source values remain available
        for context. Traversed arrays are compared at the same positional index;
        a whole-container comparator owns its internal comparison instead.
        """
        predicted, expected = json.loads(response), json.loads(ground_truth)
        contract = self.config.extraction_schema
        if contract is not None:
            contract.validate_gold(expected)
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        comparisons: JsonObject = {}
        structure = {"table_row": [0, 0, 0], "table_count": [0, 0, 0]}
        exact = True

        def present(value: Any) -> bool:
            """Distinguish absence from valid zero and false values."""
            return value is not MISSING and value is not None and value != ""

        def visit(gold: Any, prediction: Any, path: str, prediction_path: str = "", pattern: str = "") -> None:
            """Accumulate directional errors while comparing arrays positionally."""
            nonlocal exact
            if not self.config.selects_path(path, include_ancestors=True):
                return
            exact = exact and ((gold is MISSING) == (prediction is MISSING))
            field = contract.fields.get(pattern) if contract is not None else None
            if field is not None or (not isinstance(gold, (dict, list)) and not isinstance(prediction, (dict, list))):
                if field is None and not self.config.selects_path(path):
                    return
                gold_present, prediction_present = present(gold), present(prediction)
                equal = not gold_present and not prediction_present and type(gold) is type(prediction)
                reason = None
                if field is not None:
                    if gold_present:
                        model_gold = field.validator.validate_python(gold)
                    if prediction_present:
                        try:
                            model_prediction = field.validator.validate_python(prediction)
                        except ValidationError:
                            equal = False
                            reason = "Prediction violates the field's Pydantic type or constraints"
                        else:
                            if gold_present:
                                equal = field.compare(model_gold, model_prediction)
                elif gold_present and prediction_present:
                    equal = equal_values(gold, prediction)
                exact = exact and equal
                if path.startswith("/segments/") and "/fields/" not in path:
                    return
                counts[pattern][0] += int(gold_present and prediction_present and equal)
                counts[pattern][1] += int(prediction_present and not equal)
                counts[pattern][2] += int(gold_present and not equal)
                if field is not None:
                    comparisons[path] = {
                        "match": equal, "reason": reason,
                        "status": "compared" if gold_present and prediction_present else "missing" if gold_present else "extra" if prediction_present else "absent",
                        "prediction_path": prediction_path,
                    }
                return
            if gold is not MISSING and prediction is not MISSING and type(gold) is not type(prediction) and (isinstance(gold, (dict, list)) or isinstance(prediction, (dict, list))):
                visit(gold, MISSING, path, prediction_path, pattern)
                visit(MISSING, prediction, path, prediction_path, pattern)
                return
            if isinstance(gold, dict) or isinstance(prediction, dict):
                gs, ps = gold if isinstance(gold, dict) else {}, prediction if isinstance(prediction, dict) else {}
                for key in sorted(gs.keys() | ps.keys()):
                    visit(gs.get(key, MISSING), ps.get(key, MISSING), pointer(path, key), pointer(prediction_path, key), pointer(pattern, key))
                return
            if isinstance(gold, list) or isinstance(prediction, list):
                gitems, pitems = gold if isinstance(gold, list) else [], prediction if isinstance(prediction, list) else []
                exact = exact and isinstance(gold, list) and isinstance(prediction, list)
                table = self.config.tables.get(pattern)
                exact = exact and len(gitems) == len(pitems)
                for index in range(max(len(gitems), len(pitems))):
                    if not self.config.selects_path(f"{path}/{index}", include_ancestors=True):
                        continue
                    gold_item = gitems[index] if index < len(gitems) else MISSING
                    predicted_item = pitems[index] if index < len(pitems) else MISSING
                    if table:
                        count = structure[table.level]
                        count[0] += int(gold_item is not MISSING and predicted_item is not MISSING)
                        count[1] += int(gold_item is MISSING)
                        count[2] += int(predicted_item is MISSING)
                    visit(gold_item, predicted_item, f"{path}/{index}", prediction_path=f"{prediction_path}/{index}", pattern=f"{pattern}/*")
                return
        visit(expected, predicted, "")
        total = [sum(value[i] for value in counts.values()) for i in range(3)]
        result: JsonObject = {
            "tp": total[0], "fp": total[1], "fn": total[2], **ratios(*total),
            "exact_match": int(exact), "field_counts": dict(counts),
        }
        if contract is not None:
            result["comparisons"] = comparisons
        for level, count in structure.items():
            result.update({f"{level}_{name}": value for name, value in ratios(*count).items()})
            result.update({f"{level}_{name}": value for name, value in zip(("tp", "fp", "fn"), count, strict=True)})
        return result