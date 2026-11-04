"""Parse CU segments with local Markdown and normalized extraction fields.

Any describes recursively nested, caller-defined JSON values at this boundary.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..types import JsonObject

MISSING = object()
VALUE_KEYS = ("valueString", "valueNumber", "valueInteger", "valueDate", "valueTime", "valueDateTime", "valueBoolean", "valueJson", "value")


def pointer(path: str, name: str) -> str:
    """Append a properly escaped JSON Pointer token."""
    return f"{path}/{name.replace('~', '~0').replace('/', '~1')}"


@dataclass(frozen=True)
class ParsedCUSegment:
    """Keep one segment's source identity, local Markdown, and normalized fields.

    Attributes:
        content_path: Original CU content path or the declared child path for an unrouted segment.
        category: CU classification category, or None when not supplied.
        pages: Inclusive start and end page numbers, with None for missing bounds.
        markdown: Source Markdown for this segment, empty when local text is unavailable.
        fields: Unwrapped extraction values without confidence, geometry, or spans.

    Attribute references are frozen; nested field dictionaries and page lists
    remain mutable. An unrouted declaration has empty Markdown and fields.
    """

    content_path: str
    category: str | None
    pages: list[int | None]
    markdown: str
    fields: JsonObject


@dataclass(frozen=True)
class ParsedCUResult:
    """Keep segment-local Markdown and fields without collecting CU evidence.

    Attributes:
        segments: Ordered parsed segment dataclasses containing local Markdown and normalized fields.
        string_encoding: Source span encoding metadata, preserved without coercion.
    """

    segments: list[ParsedCUSegment]
    string_encoding: object


def parse_cu_result(raw: Mapping[str, Any]) -> ParsedCUResult:
    """Parse successful CU content without applying evaluation budgets.

    Args:
        raw: Raw CU dictionary, optionally wrapped in a result property.

    Returns:
        Ordered segment-local Markdown and detached JSON-compatible fields, plus encoding metadata, without mutating the input or retaining confidence, geometry, or spans.

    Raises:
        TypeError: Raw CU data is not a mapping or field values are not JSON-compatible.
        ValueError: The operation failed, its content is malformed, or values contain non-finite numbers.
    """
    if not isinstance(raw, Mapping):
        raise TypeError("cu_result must be a raw CU dictionary")
    if "status" in raw and str(raw["status"]).lower() != "succeeded":
        raise ValueError("CU operation status must be succeeded")
    payload = raw.get("result", raw)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("contents"), list):
        raise ValueError("Expected a CU result with a contents array")
    segments: list[ParsedCUSegment] = []
    declared: list[tuple[str, JsonObject]] = []

    def field_value(field: Any, path: str) -> Any:
        """Unwrap a typed field without copying confidence, geometry, or spans."""
        if field is None:
            return None
        if not isinstance(field, dict):
            raise ValueError(f"Malformed CU field at {path}")
        if "valueObject" in field:
            return None if field["valueObject"] is None else object_value(field["valueObject"], path)
        if "valueArray" in field:
            if field["valueArray"] is None:
                return None
            if not isinstance(field["valueArray"], list):
                raise ValueError(f"Expected a CU array at {path}")
            return [field_value(child, f"{path}/{index}") for index, child in enumerate(field["valueArray"])]
        for key in VALUE_KEYS:
            if key in field:
                return json.loads(json.dumps(field[key], ensure_ascii=False, allow_nan=False))
        if any(isinstance(key, str) and key.startswith("value") for key in field):
            raise ValueError(f"Unsupported CU value type at {path}")
        return None

    def object_value(fields: Any, path: str) -> JsonObject:
        """Preserve source field names while unwrapping their values."""
        if not isinstance(fields, dict):
            raise ValueError(f"Expected a CU object at {path}")
        result: JsonObject = {}
        for name, field in fields.items():
            if not isinstance(name, str):
                raise ValueError("CU object keys must be strings")
            result[name] = field_value(field, pointer(path, name))
        return result

    seen_paths: set[str] = set()
    content_paths: set[str] = set()
    for content_index, content in enumerate(payload["contents"]):
        if not isinstance(content, dict):
            raise ValueError("Every CU content must be an object")
        content_path = content.get("path", f"content-{content_index}")
        if not isinstance(content_path, str) or not content_path:
            raise ValueError("CU content path must be a nonempty string")
        if content_path in seen_paths:
            raise ValueError("Duplicate CU content path")
        seen_paths.add(content_path)
        text = content.get("markdown", "")
        if text is not None and not isinstance(text, str):
            raise ValueError("CU Markdown must be a string or null")
        if content.get("segments"):
            if not isinstance(content["segments"], list) or not all(isinstance(item, dict) for item in content["segments"]):
                raise ValueError("CU segments must be an array of objects")
            declared.extend((content_path, item) for item in content["segments"])
            continue
        content_paths.add(content_path)
        path = f"/segments/{len(segments)}/fields"
        fields = content.get("fields")
        segments.append(ParsedCUSegment(content_path=content_path, category=content.get("category"), pages=[content.get("startPageNumber"), content.get("endPageNumber")], markdown=text or "", fields=object_value({} if fields is None else fields, path)))
    for parent, segment in declared:
        if "segmentId" not in segment:
            raise ValueError("Declared CU segments require segmentId")
        child_path = f"{parent}/{segment['segmentId']}"
        if child_path not in content_paths:
            segments.append(ParsedCUSegment(content_path=child_path, category=segment.get("category"), pages=[segment.get("startPageNumber"), segment.get("endPageNumber")], markdown="", fields={}))
    if not segments:
        raise ValueError("CU returned no analyzable document segments")
    return ParsedCUResult(segments=segments, string_encoding=payload.get("stringEncoding", raw.get("stringEncoding", "codePoint")))