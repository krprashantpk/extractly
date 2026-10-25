"""Score extraction occurrences rather than substituting text similarity.

Any is required for recursive values in caller-defined extraction schemas.
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any

from pydantic import ValidationError

from .comparison import equal_values
from .config import EvaluationConfig
from .cu.cu_result_parser import MISSING, pointer, slots
from .types import JsonObject


def ratios(tp: int, fp: int, fn: int) -> dict[str, float | None]:
    """Calculate precision, recall, and F1 from positive occurrence counts.

    Precision measures the fraction of predicted occurrences that are correct.
    Recall measures the fraction of expected occurrences captured correctly.
    F1 balances precision and recall using their harmonic mean.
    For field-value scoring, a wrong nonempty value contributes both a false positive and a false negative.
    The caller must supply nonnegative counts; this function does not validate them.

    Args:
        tp: Number of correctly matched positive occurrences, called true positives.
        fp: Number of incorrect or unexpected predicted occurrences, called false positives.
        fn: Number of expected occurrences missing or predicted incorrectly, called false negatives.

    Returns:
        A dictionary containing precision, recall, and f1 scores between zero and one for nonnegative counts; each score is None when its denominator is zero, never an assumed perfect score.
    """
    return {
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


class SchemaEvaluator:
    """Validate original canonical values using the declared extraction model."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Accept the caller's single extraction schema contract."""
        self.config = config

    def __call__(self, *, response: str, **kwargs: object) -> JsonObject:
        """Return conformance, violation count, and escaped violation paths."""
        value = json.loads(response)
        paths = self.config.model_contract.violation_paths(value) if self.config.model_contract is not None else []
        return {"valid": int(not paths), "violations": len(paths), "paths": paths}


class ExtractionEvaluator:
    """Compute typed field and occurrence-aware row/document-instance metrics."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Accept explicit normalization and independent identity rules."""
        self.config = config

    def __call__(self, *, response: str, ground_truth: str, **kwargs: object) -> JsonObject:
        """Compare extraction against independent gold, preserving duplicate counts.

        Wrong nonempty values contribute both a false positive and a false
        negative. Invalid predictions count as errors; invalid gold raises.
        Category and page metadata affect exact match but not field F1.
        With extraction_schema, only declared fields and their containing structure
        affect scores and exact match; undeclared source values remain available
        for row alignment and context.
        """
        predicted, expected = json.loads(response), json.loads(ground_truth)
        contract = self.config.model_contract
        if contract is not None:
            contract.validate_gold(expected)
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        tolerant: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        comparisons: JsonObject = {}
        structure = {"row": [0, 0, 0], "instance": [0, 0, 0]}
        exact = True

        def present(value: Any) -> bool:
            """Distinguish absence from valid zero and false values."""
            return value is not MISSING and value is not None and value != ""

        def visit(gold: Any, prediction: Any, path: str, prediction_path: str = "", pattern: str = "") -> None:
            """Accumulate directional errors while independently aligning arrays."""
            nonlocal exact
            if not self.config.selects_path(path, include_ancestors=True):
                return
            exact = exact and ((gold is MISSING) == (prediction is MISSING))
            if contract is not None and pattern in contract.fields:
                field = contract.fields[pattern]
                gold_present, prediction_present = present(gold), present(prediction)
                equal = relaxed = not gold_present and not prediction_present and type(gold) is type(prediction)
                reason = None
                if gold_present:
                    model_gold = field.validator.validate_python(gold)
                if prediction_present:
                    try:
                        model_prediction = field.validator.validate_python(prediction)
                    except ValidationError:
                        equal = relaxed = False
                        reason = "Prediction violates the field's Pydantic type or constraints"
                    else:
                        if gold_present:
                            comparison = field.compare(model_gold, model_prediction)
                            equal, relaxed, reason = comparison.exact, comparison.tolerant, comparison.reason
                exact = exact and equal
                for target, match in ((counts, equal), (tolerant, relaxed)):
                    target[pattern][0] += int(gold_present and prediction_present and match)
                    target[pattern][1] += int(prediction_present and not match)
                    target[pattern][2] += int(gold_present and not match)
                comparisons[path] = {
                    "exact": equal, "tolerant": relaxed, "reason": reason,
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
                pairs = []
                if table and table.keys:
                    keys = tuple(table.keys)

                    def identity(row: Any) -> str | None:
                        """Match by independent source identities without dropping duplicates."""
                        values = [slots(row, key if key.startswith("/") else pointer("", key))[0][1] for key in keys]
                        return json.dumps(values, sort_keys=True, allow_nan=False) if all(present(v) for v in values) else None

                    queue: dict[str, deque[int]] = defaultdict(deque)
                    for i, row in enumerate(pitems):
                        key = identity(row)
                        if key is not None:
                            queue[key].append(i)
                    used = set()
                    for gi, row in enumerate(gitems):
                        key = identity(row)
                        if key is None:
                            raise ValueError(f"Gold table lacks an independent identity at {path}")
                        index = queue[key].popleft() if queue[key] else None
                        if index is not None:
                            used.add(index)
                        pairs.append((row, pitems[index] if index is not None else MISSING, gi, index))
                    pairs.extend((MISSING, row, i, i) for i, row in enumerate(pitems) if i not in used)
                else:
                    pairs = [(gitems[i] if i < len(gitems) else MISSING, pitems[i] if i < len(pitems) else MISSING, i, i) for i in range(max(len(gitems), len(pitems)))]
                exact = exact and len(gitems) == len(pitems)
                for g, p, gi, pi in pairs:
                    if not self.config.selects_path(f"{path}/{gi}", include_ancestors=True):
                        continue
                    if table:
                        count = structure[table.level]
                        count[0] += int(g is not MISSING and p is not MISSING)
                        count[1] += int(g is MISSING)
                        count[2] += int(p is MISSING)
                    visit(g, p, f"{path}/{gi}", prediction_path=f"{prediction_path}/{pi}", pattern=f"{pattern}/*")
                return
            if not self.config.selects_path(path):
                return
            g, p = present(gold), present(prediction)
            equal = relaxed = not g and not p and type(gold) is type(prediction)
            if g and p:
                comparison = equal_values(gold, prediction)
                equal, relaxed = comparison.exact, comparison.tolerant
            exact = exact and equal
            if path.startswith("/segments/") and "/fields/" not in path:
                return
            for target, match in ((counts, equal), (tolerant, relaxed)):
                target[pattern][0] += int(g and p and match)
                target[pattern][1] += int(p and not match)
                target[pattern][2] += int(g and not match)

        visit(expected, predicted, "")
        total = [sum(value[i] for value in counts.values()) for i in range(3)]
        loose = [sum(value[i] for value in tolerant.values()) for i in range(3)]
        result: JsonObject = {
            "tp": total[0], "fp": total[1], "fn": total[2], **ratios(*total),
            "exact_match": int(exact), "tolerant": ratios(*loose),
            "field_counts": dict(counts), "tolerant_field_counts": dict(tolerant),
        }
        if contract is not None:
            result["comparisons"] = comparisons
        for level, count in structure.items():
            result.update({f"{level}_{name}": value for name, value in ratios(*count).items()})
            result.update({f"{level}_{name}": value for name, value in zip(("tp", "fp", "fn"), count, strict=True)})
        return result
