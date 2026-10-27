"""Calculate shared occurrence-based evaluation metrics."""


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
        A dictionary containing precision, recall, and f1 scores rounded to three decimal places between zero and one for nonnegative counts; each score is None when its denominator is zero, never an assumed perfect score.

    Examples:
        | Field    | Expected | Extracted | Count              |
        |----------|----------|-----------|--------------------|
        | Name     | Alice    | Alice     | tp + 1             |
        | Quantity | 10       | 12        | fp + 1, fn + 1     |
        | City     | Delhi    | Missing   | fn + 1             |

        Totals are tp = 1, fp = 1, and fn = 2. True positives count correct field occurrences, not entire documents.
    """
    return {
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "f1": round(2 * tp / (2 * tp + fp + fn), 3) if 2 * tp + fp + fn else None,
    }