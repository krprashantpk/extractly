"""Score extraction occurrences rather than substituting text similarity.

Any is required for recursive values in caller-defined extraction schemas.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict, deque
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation, localcontext
from typing import Any

from jsonschema import FormatChecker
from jsonschema.validators import validator_for

from .config import EvaluationConfig, FieldRule
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
    """Validate canonical shape and required field occurrences with JSON Schema."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Build a validator from caller-supplied schema and field policy."""
        self.validator = validator_for(config.extraction_schema)(config.extraction_schema, format_checker=FormatChecker())
        self.required = [pattern for pattern, rule in config.fields.items() if rule.required]

    def __call__(self, *, response: str, **kwargs: object) -> JsonObject:
        """Return conformance, violation count, and escaped violation paths."""
        value = json.loads(response)
        paths = []
        for error in self.validator.iter_errors(value):
            path = ""
            for part in error.path:
                path = pointer(path, str(part))
            paths.append(path)
        paths.extend(path for pattern in self.required for path, found in slots(value, pattern) if found is MISSING or found is None or found == "")
        return {"valid": int(not paths), "violations": len(paths), "paths": sorted(set(paths))}


class ExtractionEvaluator:
    """Compute typed field and occurrence-aware row/document-instance metrics."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Accept explicit normalization and independent identity rules."""
        self.config = config

    @staticmethod
    def _normalized(value: Any, rule: FieldRule, parent: Any, root: Any, path: str) -> Any:
        """Normalize only when an explicit field policy permits it."""
        if rule.kind == "decimal":
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError("Expected a decimal quantity")
            if isinstance(value, str) and not rule.coerce_numeric_strings:
                raise ValueError("Numeric strings require an explicit coercion policy")
            number = Decimal(str(value))
            if not number.is_finite():
                raise ValueError("Decimal values must be finite")
            if rule.unit_field:
                if rule.unit_field.startswith("/"):
                    indices = iter(re.findall(r"/(\d+)(?=/|$)", path))
                    reference = "/".join(next(indices) if part == "*" else part for part in rule.unit_field.split("/"))
                    unit = slots(root, reference)[0][1]
                else:
                    unit = parent.get(rule.unit_field) if isinstance(parent, dict) else None
                if not isinstance(unit, str) or unit not in rule.unit_scales:
                    raise ValueError("Quantity unit has no declared conversion")
                factor = Decimal(rule.unit_scales[unit])
                with localcontext() as context:
                    context.prec = max(28, len(number.as_tuple().digits) + len(factor.as_tuple().digits))
                    number *= factor
            return number
        if rule.kind == "date":
            if not isinstance(value, str):
                raise ValueError("Expected a date string")
            return datetime.strptime(value, rule.date_format).replace(tzinfo=UTC).date()
        if rule.kind == "text":
            if not isinstance(value, str):
                raise ValueError("Expected text")
            return " ".join(unicodedata.normalize("NFC", value).split())
        return value

    def __call__(self, *, response: str, ground_truth: str, **kwargs: object) -> JsonObject:
        """Compare extraction against independent gold, preserving duplicate counts.

        Wrong nonempty values contribute both a false positive and a false
        negative. Invalid predictions count as errors; invalid gold raises.
        Category and page metadata affect exact match but not field F1.
        """
        predicted, expected = json.loads(response), json.loads(ground_truth)
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        tolerant: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        structure = {"row": [0, 0, 0], "instance": [0, 0, 0]}
        exact = True

        def present(value: Any) -> bool:
            """Distinguish absence from valid zero and false values."""
            return value is not MISSING and value is not None and value != ""

        def visit(
            gold: Any,
            prediction: Any,
            path: str,
            gp: Any = None,
            pp: Any = None,
            prediction_path: str = "",
            pattern: str = "",
        ) -> None:
            """Accumulate directional errors while independently aligning arrays."""
            nonlocal exact
            exact = exact and ((gold is MISSING) == (prediction is MISSING))
            if gold is not MISSING and prediction is not MISSING and type(gold) is not type(prediction) and (isinstance(gold, (dict, list)) or isinstance(prediction, (dict, list))):
                visit(gold, MISSING, path, gp, pp, prediction_path, pattern)
                visit(MISSING, prediction, path, gp, pp, prediction_path, pattern)
                return
            if isinstance(gold, dict) or isinstance(prediction, dict):
                gs, ps = gold if isinstance(gold, dict) else {}, prediction if isinstance(prediction, dict) else {}
                for key in sorted(gs.keys() | ps.keys()):
                    visit(gs.get(key, MISSING), ps.get(key, MISSING), pointer(path, key), gs, ps, pointer(prediction_path, key), pointer(pattern, key))
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
                    if table:
                        count = structure[table.level]
                        count[0] += int(g is not MISSING and p is not MISSING)
                        count[1] += int(g is MISSING)
                        count[2] += int(p is MISSING)
                    visit(g, p, f"{path}/{gi}", prediction_path=f"{prediction_path}/{pi}", pattern=f"{pattern}/*")
                return
            rule = self.config.fields.get(pattern, FieldRule())
            g, p = present(gold), present(prediction)
            equal = relaxed = not g and not p and type(gold) is type(prediction)
            if g:
                normalized_gold = self._normalized(gold, rule, gp, expected, path)
            if g and p:
                try:
                    normalized_prediction = self._normalized(prediction, rule, pp, predicted, prediction_path)
                    equal = normalized_gold == normalized_prediction and (isinstance(gold, bool) == isinstance(prediction, bool))
                    relaxed = equal
                    if rule.kind == "decimal":
                        with localcontext() as context:
                            context.prec = max(28, len(normalized_gold.as_tuple().digits) + len(normalized_prediction.as_tuple().digits) + len(Decimal(rule.relative_tolerance).as_tuple().digits))
                            relaxed = abs(normalized_gold - normalized_prediction) <= max(Decimal(rule.absolute_tolerance), Decimal(rule.relative_tolerance) * abs(normalized_gold))
                except (ValueError, TypeError, InvalidOperation):
                    equal = relaxed = False
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
        for level, count in structure.items():
            result.update({f"{level}_{name}": value for name, value in ratios(*count).items()})
            result.update({f"{level}_{name}": value for name, value in zip(("tp", "fp", "fn"), count, strict=True)})
        return result
