"""Compare normalized extraction fields against reference values without configuration."""

from __future__ import annotations

import logging
from collections import defaultdict

from pydantic import ValidationError

from ..comparison import EvaluationModel, ModelContract, ModelField
from ..cu.cu_result_parser import MISSING, pointer
from ..types import JsonObject
from ._shared import ratios

logger = logging.getLogger(__name__)


class FieldEvaluator:
    """Score model-selected field occurrences independently of table structure.

    Inputs are normalized dictionaries, not canonical segment wrappers or JSON
    strings. The caller supplies the model on every call. Registered comparators
    receive Pydantic-validated values; original inputs are not repaired or filled
    with defaults. Arrays are positional unless a whole-container comparator
    owns that field. Missing and extra rows contribute missing and extra field
    occurrences, but no table or row metrics are calculated here.

    Missing keys, null, and empty strings are absent; zero and false are present.
    Wrong present values count as both FP and FN. Ratios are rounded to three
    decimal places and are None when their denominator is zero.

    Per-document field metrics:
        extraction.document.tp: Present expected and extracted field-value occurrences accepted by the configured comparison policy.
        extraction.document.fp: Incorrect or extra present extracted field-value occurrences, including invalid predicted types or constraints.
        extraction.document.fn: Present expected field-value occurrences that are missing, invalid, or predicted incorrectly.
        extraction.document.precision: tp / (tp + fp), the fraction of extracted field-value occurrences that are correct.
        extraction.document.recall: tp / (tp + fn), the fraction of expected field-value occurrences captured correctly.
        extraction.document.f1: 2 * tp / (2 * tp + fp + fn), the harmonic balance of precision and recall.
    """

    def __call__(self, extracted_values: JsonObject, ground_truth: JsonObject, extraction_model: type[EvaluationModel]) -> JsonObject:
        """Return field metrics, occurrence counts, and concrete comparison diagnostics.

        Missing keys, null, and empty strings are absent; zero and false are
        present. A wrong present value contributes both FP and FN. Undefined
        ratios are None; defined ratios are rounded to three decimal places.
        Exact match describes field judgments only, not structural agreement.

        Args:
            extracted_values: Normalized candidate fields retaining original nesting and array order.
            ground_truth: Reference fields whose present values and requiredness satisfy the model.
            extraction_model: EvaluationModel class defining field scope, validation, and comparisons.

        Returns:
            Metrics with extraction.document prefixes, field-only exact_match, wildcard field_counts, and comparisons keyed by escaped concrete JSON Pointers.

        Raises:
            TypeError: Inputs have unsupported types or a comparator does not return bool.
            ValueError: The model has empty, recursive, or otherwise unsupported declarations.
            ValidationError: Ground truth violates the model's types or constraints.

        Caller comparator exceptions propagate unchanged. No clients, configuration,
        source Markdown, or table rules are needed or retained between calls.
        """
        if not isinstance(extracted_values, dict) or not isinstance(ground_truth, dict):
            raise TypeError("extracted_values and ground_truth must be dictionaries")
        if not isinstance(extraction_model, type) or not issubclass(extraction_model, EvaluationModel):
            raise TypeError("extraction_model must be an EvaluationModel class")
        contract = ModelContract.compile(extraction_model)
        extraction_model.model_validate(ground_truth)
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
        comparisons: JsonObject = {}
        logger.info("Starting field evaluation with model '%s' for %s field patterns", extraction_model.__name__, len(contract.fields))

        def present(value: object) -> bool:
            """Distinguish absent values from valid zero and false values."""
            return value is not MISSING and value is not None and value != ""

        def visit(expected: object, actual: object, parts: tuple[str, ...], path: str, pattern: str, field: ModelField) -> None:
            """Expand one model field into positional occurrences without filling defaults."""
            if parts:
                name, remaining = parts[0], parts[1:]
                if name == "*":
                    expected_items = expected if isinstance(expected, list) else []
                    actual_items = actual if isinstance(actual, list) else []
                    for index in range(max(len(expected_items), len(actual_items))):
                        expected_item = expected_items[index] if index < len(expected_items) else MISSING
                        actual_item = actual_items[index] if index < len(actual_items) else MISSING
                        visit(expected_item, actual_item, remaining, pointer(path, str(index)), pattern, field)
                else:
                    expected_value = expected.get(name, MISSING) if isinstance(expected, dict) else MISSING
                    actual_value = actual.get(name, MISSING) if isinstance(actual, dict) else MISSING
                    visit(expected_value, actual_value, remaining, pointer(path, name), pattern, field)
                return
            expected_present, actual_present = present(expected), present(actual)
            equal = not expected_present and not actual_present
            reason = None
            if expected_present:
                validated_expected = field.validator.validate_python(expected)
            if actual_present:
                try:
                    validated_actual = field.validator.validate_python(actual)
                except ValidationError:
                    reason = "Prediction violates the field's Pydantic type or constraints"
                else:
                    if expected_present:
                        equal = field.compare(validated_expected, validated_actual)
            counts[pattern][0] += int(expected_present and actual_present and equal)
            counts[pattern][1] += int(actual_present and not equal)
            counts[pattern][2] += int(expected_present and not equal)
            comparisons[path] = {
                "match": equal,
                "reason": reason,
                "status": "compared" if expected_present and actual_present else "missing" if expected_present else "extra" if actual_present else "absent",
                "prediction_path": path,
            }

        for canonical_pattern, field in contract.fields.items():
            pattern = canonical_pattern.removeprefix("/segments/*/fields")
            parts = tuple(part.replace("~1", "/").replace("~0", "~") for part in pattern.split("/")[1:])
            visit(ground_truth, extracted_values, parts, "", pattern, field)
        tp, fp, fn = (sum(count[index] for count in counts.values()) for index in range(3))
        logger.info("Completed field evaluation with model '%s': TP=%s FP=%s FN=%s", extraction_model.__name__, tp, fp, fn)
        return {
            "metrics": {f"extraction.document.{name}": value for name, value in ({"tp": tp, "fp": fp, "fn": fn} | ratios(tp, fp, fn)).items()},
            "exact_match": int(all(comparison["match"] for comparison in comparisons.values())),
            "field_counts": dict(counts),
            "comparisons": comparisons,
        }