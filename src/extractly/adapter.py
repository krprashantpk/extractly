"""Adapt CU wire JSON without importing an application's extraction models.

Any describes recursively nested, caller-defined JSON values at this boundary.
Raw values and evidence are preserved without business-specific defaults.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .config import EvaluationConfig
from .types import JsonObject

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
        return [slot for i, child in enumerate(value if isinstance(value, list) else []) for slot in slots(child, tail, f"{path}/{i}")]
    child = (
        value.get(part, MISSING)
        if isinstance(value, dict)
        else value[int(part)]
        if isinstance(value, list) and part.isdigit() and int(part) < len(value)
        else MISSING
    )
    return slots(child, tail, pointer(path, part))


class CUAdapter:
    """Preserve typed values, segment occurrences, and field-level evidence."""

    def __init__(self, config: EvaluationConfig) -> None:
        """Accept the caller's validated evaluation contract."""
        if not isinstance(config, EvaluationConfig):
            raise TypeError("config must be an EvaluationConfig")
        self.config = config

    def prepare(self, raw: Mapping[str, Any], document_id: str) -> JsonObject:
        """Convert a successful CU response into a Foundry input row.

        Args:
            raw: Raw CU dictionary, optionally wrapped in a result property.
            document_id: Nonempty caller-owned source identity.

        Returns:
            Canonical response, full Markdown context, and preserved evidence metadata.

        Raises:
            TypeError: An input has an unsupported type.
            ValueError: CU data is malformed or context exceeds its budget.
        """
        if not isinstance(raw, Mapping):
            raise TypeError("cu_result must be a raw CU dictionary")
        if not isinstance(document_id, str):
            raise TypeError("document_id must be a string")
        if not document_id.strip():
            raise ValueError("document_id must not be blank")
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
                return [field_value(child, f"{path}/{i}", content_path) for i, child in enumerate(field["valueArray"])]
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
        context = "\n\n".join(f"[{path}]\n{text}" for path, text in markdown.items())
        if len(context) > self.config.max_context_characters:
            raise ValueError("Document exceeds configured context budget; split it explicitly, never truncate")
        return {
            "id": document_id,
            "response": json.dumps({"segments": segments}, ensure_ascii=False, allow_nan=False),
            "context": context,
            "query": "Evaluate the extracted document fields and all row occurrences.",
            "evidence": evidence,
            "markdown": markdown,
            "string_encoding": payload.get("stringEncoding", raw.get("stringEncoding", "codePoint")),
        }

    def field_rows(self, record: JsonObject, *, max_fields_per_document: int = 2000) -> list[JsonObject]:
        """Include emitted leaves and required absences without exposing CU confidence.

        Unknown omitted rows require the separate completeness evaluator.
        The caller supplies the online field-grading budget explicitly.
        """
        if not isinstance(max_fields_per_document, int) or isinstance(max_fields_per_document, bool):
            raise TypeError("max_fields_per_document must be an integer")
        if max_fields_per_document < 1:
            raise ValueError("max_fields_per_document must be positive")
        response = json.loads(record["response"])
        emitted: JsonObject = {}

        def walk(value: Any, path: str = "") -> None:
            """Collect scalar field values while excluding segment metadata."""
            if isinstance(value, dict):
                for key, child in value.items():
                    walk(child, pointer(path, key))
            elif isinstance(value, list):
                for i, child in enumerate(value):
                    walk(child, f"{path}/{i}")
            elif "/fields/" in path:
                emitted[path] = value

        walk(response)
        for pattern, rule in self.config.fields.items():
            emitted.update((path, value) for path, value in slots(response, pattern) if rule.required or value is not MISSING)
        if len(emitted) > max_fields_per_document:
            raise ValueError("Document exceeds configured field-grading budget")

        def scope(path: str) -> JsonObject:
            """Supply the field's row and document instance context."""
            parent = path.rsplit("/", 1)[0]
            instance = slots(response, "/".join(path.split("/")[:3]))[0][1]
            if not isinstance(instance, dict) or not isinstance(instance.get("fields"), dict):
                raise ValueError("Field grading rules must address slots within canonical document segments")
            return {
                "row_context": json.dumps(slots(response, parent)[0][1]) if parent.split("/")[-1].isdigit() else "{}",
                "instance": json.dumps({
                    "category": instance["category"],
                    "pages": instance["pages"],
                    "fields": {name: value for name, value in instance["fields"].items() if not isinstance(value, (dict, list))},
                }),
            }

        return [{
            "id": f"{len(record['id'])}:{record['id']}{path}",
            "document_id": record["id"],
            "field_path": path,
            "response": json.dumps({"presence": "absent"} if value is MISSING else {"value": value}, ensure_ascii=False, allow_nan=False),
            "context": record["context"],
            "query": f"Assess only the contract slot {path}.",
            "contract": self.config.model_dump_json(include={"fields", "tables"}),
            **scope(path),
        } for path, value in emitted.items()]
