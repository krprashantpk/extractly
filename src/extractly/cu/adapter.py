"""Adapt canonical CU values and evidence into evaluation records.

Any describes recursively nested, caller-defined JSON values at this boundary.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from ..config import EvaluationConfig
from ..types import JsonObject
from .cu_result_parser import MISSING, parse_cu_result, pointer, slots


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
        parsed = parse_cu_result(raw)
        context = "\n\n".join(f"[{path}]\n{text}" for path, text in parsed.markdown.items())
        if len(context) > self.config.max_context_characters:
            raise ValueError("Document exceeds configured context budget; split it explicitly, never truncate")
        return {
            "id": document_id,
            "response": json.dumps(parsed.response, ensure_ascii=False, allow_nan=False),
            "context": context,
            "query": "Evaluate the extracted document fields and all row occurrences.",
            "evidence": parsed.evidence,
            "markdown": parsed.markdown,
            "string_encoding": parsed.string_encoding,
        }

    def field_rows(self, record: JsonObject, *, max_fields_per_document: int = 2000) -> list[JsonObject]:
        """Include emitted leaves and required absences without exposing CU confidence.

        extraction_model selects graded slots before the field budget is checked.
        Full response, row context, and evidence remain available for grounding.
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
                for index, child in enumerate(value):
                    walk(child, f"{path}/{index}")
            elif "/fields/" in path:
                emitted[path] = value

        if self.config.extraction_schema is not None:
            violations = set(self.config.extraction_schema.violation_paths(response))
            for pattern in self.config.extraction_schema.fields:
                emitted.update((path, value) for path, value in slots(response, pattern) if value is not MISSING or path in violations)
        else:
            walk(response)
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
            "contract": self.config.model_dump_json(include={"extraction_model", "tables"}),
            **scope(path),
        } for path, value in emitted.items()]