"""Build editable annotation drafts, not independently verified gold, from CU.

Any describes recursively nested, caller-defined JSON values at this boundary.
"""

import json
from collections.abc import Mapping
from typing import Any

from ..types import JsonObject
from .cu_result_parser import parse_cu_result


def build_ground_truth(raw: Mapping[str, Any]) -> JsonObject:
    """Build a canonical ground-truth draft for independent source review.

    CU predictions are not verified annotations. Correct all values and add
    missing fields, rows, and segments against the original source before
    using the result as offline ground truth. No evaluation configuration,
    service calls, or file writes are involved.

    Args:
        raw: Raw CU dictionary, optionally wrapped in a successful result envelope.

    Returns:
        A detached JSON-compatible dictionary containing segments with category, pages, and unwrapped fields; editing it does not mutate raw.

    Raises:
        TypeError: Raw CU data has an unsupported type or contains non-JSON values.
        ValueError: CU content is malformed, the operation failed, or canonical values contain non-finite numbers.
    """
    draft: JsonObject = json.loads(json.dumps(parse_cu_result(raw).response, ensure_ascii=False, allow_nan=False))
    return draft