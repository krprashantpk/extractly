"""Parse CU values and source associations for adaptation and annotation drafts.

Any describes recursively nested, caller-defined JSON values at this boundary.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..types import JsonObject

MISSING = object()
VALUE_KEYS = ("valueString", "valueNumber", "valueInteger", "valueDate", "valueTime", "valueDateTime", "valueBoolean", "value")


def pointer(path: str, name: str) -> str:
    """Append a properly escaped JSON Pointer token."""
    return f"{path}/{name.replace('~', '~0').replace('/', '~1')}"


def slots(value: Any, pattern: str, path: str = "") -> list[tuple[str, Any]]:
    """Enumerate array occurrences and explicitly absent configured leaf slots."""
    if not pattern:
        return [(path, value)]
    part, _, rest = pattern.lstrip("/").partition("/")
    part = part.replace("~1", "/").replace("~0", "~")
    tail = f"/{rest}" if rest else ""
    if part == "*":
        return [slot for index, child in enumerate(value if isinstance(value, list) else []) for slot in slots(child, tail, f"{path}/{index}")]
    child = (
        value.get(part, MISSING)
        if isinstance(value, dict)
        else value[int(part)]
        if isinstance(value, list) and part.isdigit() and int(part) < len(value)
        else MISSING
    )
    return slots(child, tail, pointer(path, part))


@dataclass(frozen=True)
class ParsedCUResult:
    """Keep canonical values separate from source evidence and Markdown.

    Attributes:
        response: Canonical segments containing unwrapped extraction values.
        evidence: CU metadata indexed by escaped canonical field paths.
        markdown: Original source text indexed by CU content path.
        string_encoding: Source span encoding metadata, preserved without coercion.
    """

    response: JsonObject
    evidence: JsonObject
    markdown: dict[str, str]
    string_encoding: object


def parse_cu_result(raw: Mapping[str, Any]) -> ParsedCUResult:
    """Parse successful CU content without applying evaluation budgets.

    Args:
        raw: Raw CU dictionary, optionally wrapped in a result property.

    Returns:
        Canonical values and source metadata without mutating the input.

    Raises:
        TypeError: Raw CU data is not a mapping.
        ValueError: The operation failed or its content is malformed.
    """
    if not isinstance(raw, Mapping):
        raise TypeError("cu_result must be a raw CU dictionary")
    if "status" in raw and str(raw["status"]).lower() != "succeeded":
        raise ValueError("CU operation status must be succeeded")
    payload = raw.get("result", raw)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("contents"), list):
        raise ValueError("Expected a CU result with a contents array")
    evidence: JsonObject = {}
    markdown: dict[str, str] = {}
    segments: list[JsonObject] = []
    declared: list[tuple[str, JsonObject]] = []

    def field_value(field: Any, path: str, content_path: str) -> Any:
        """Unwrap a typed field while retaining its source association."""
        if field is None:
            return None
        if not isinstance(field, dict):
            raise ValueError(f"Malformed CU field at {path}")
        evidence[path] = {"content_path": content_path, **{key: field[key] for key in ("type", "confidence", "source", "spans") if key in field}}
        if "valueObject" in field:
            return None if field["valueObject"] is None else object_value(field["valueObject"], path, content_path)
        if "valueArray" in field:
            if field["valueArray"] is None:
                return None
            if not isinstance(field["valueArray"], list):
                raise ValueError(f"Expected a CU array at {path}")
            return [field_value(child, f"{path}/{index}", content_path) for index, child in enumerate(field["valueArray"])]
        for key in VALUE_KEYS:
            if key in field:
                return field[key]
        if any(isinstance(key, str) and key.startswith("value") for key in field):
            raise ValueError(f"Unsupported CU value type at {path}")
        return None

    def object_value(fields: Any, path: str, content_path: str) -> JsonObject:
        """Preserve source field names while unwrapping their values."""
        if not isinstance(fields, dict):
            raise ValueError(f"Expected a CU object at {path}")
        result: JsonObject = {}
        for name, field in fields.items():
            if not isinstance(name, str):
                raise ValueError("CU object keys must be strings")
            result[name] = field_value(field, pointer(path, name), content_path)
        return result

    content_paths: set[str] = set()
    for content in payload["contents"]:
        if not isinstance(content, dict):
            raise ValueError("Every CU content must be an object")
        content_path = content.get("path", f"content-{len(markdown)}")
        if not isinstance(content_path, str) or not content_path:
            raise ValueError("CU content path must be a nonempty string")
        if content_path in markdown:
            raise ValueError("Duplicate CU content path")
        text = content.get("markdown", "")
        if text is not None and not isinstance(text, str):
            raise ValueError("CU Markdown must be a string or null")
        markdown[content_path] = text or ""
        if content.get("segments"):
            if not isinstance(content["segments"], list) or not all(isinstance(item, dict) for item in content["segments"]):
                raise ValueError("CU segments must be an array of objects")
            declared.extend((content_path, item) for item in content["segments"])
            continue
        content_paths.add(content_path)
        path = f"/segments/{len(segments)}/fields"
        fields = content.get("fields")
        segments.append({
            "category": content.get("category"),
            "pages": [content.get("startPageNumber"), content.get("endPageNumber")],
            "fields": object_value({} if fields is None else fields, path, content_path),
        })
    for parent, segment in declared:
        if "segmentId" not in segment:
            raise ValueError("Declared CU segments require segmentId")
        child_path = f"{parent}/{segment['segmentId']}"
        if child_path not in content_paths:
            evidence[f"/segments/{len(segments)}"] = {"content_path": parent, "spans": [segment["span"]] if segment.get("span") else []}
            segments.append({
                "category": segment.get("category"),
                "pages": [segment.get("startPageNumber"), segment.get("endPageNumber")],
                "fields": {},
            })
    if not segments:
        raise ValueError("CU returned no analyzable document segments")
    return ParsedCUResult(response={"segments": segments}, evidence=evidence, markdown=markdown, string_encoding=payload.get("stringEncoding", raw.get("stringEncoding", "codePoint")))