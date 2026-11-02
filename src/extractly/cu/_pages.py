"""Prepare notebook-aligned physical pages without guessing evidence boundaries."""

import re
from itertools import pairwise

from ..config import EvaluationConfig
from ..types import JsonObject
from ._analyzer import analyzer_schema, collect_targets, cu_value, source_pages
from .cu_result_parser import parse_cu_result, pointer, slots

DOCUMENT_LEVEL_FIELDS = ("TotalNetTonnage", "deliveryMode")


def page_locations(contents: list[JsonObject]) -> list[tuple[int, int]]:
    """Reject overlapping physical ranges and ambiguous or overlapping page spans."""
    locations: list[tuple[int, int]] = []
    seen: set[int] = set()
    for index, content in enumerate(contents):
        start, end = content.get("startPageNumber"), content.get("endPageNumber")
        if not isinstance(start, int) or isinstance(start, bool) or not isinstance(end, int) or isinstance(end, bool) or not 1 <= start <= end:
            raise ValueError("Every content block must have a valid page range")
        for number in range(start, end + 1):
            if number in seen:
                raise ValueError("Content page ranges overlap")
            seen.add(number)
            locations.append((index, number))
        if start != end:
            occupied: list[tuple[int, int]] = []
            for page in content.get("pages", []):
                if page.get("pageNumber") not in range(start, end + 1):
                    raise ValueError("Page metadata is outside the content page range")
                for span in page.get("spans", []):
                    offset, length = span.get("offset", 0), span.get("length")
                    if any(isinstance(value, bool) or not isinstance(value, int) for value in (offset, length)) or offset < 0 or length <= 0:
                        raise ValueError("Page span offsets and lengths must be valid integers")
                    occupied.append((offset, offset + length))
            occupied.sort()
            if any(right[0] < left[1] for left, right in pairwise(occupied)):
                raise ValueError("Spans from different pages overlap")
    return sorted(locations, key=lambda location: location[1])


def page_markdown(content: JsonObject, number: int, encoding: object) -> str:
    """Slice only validated code-point spans; single-page evidence stays unchanged."""
    markdown = content.get("markdown")
    if content.get("kind", "document") != "document" or not isinstance(markdown, str) or not markdown.strip():
        raise ValueError("Selected content must contain nonempty document Markdown")
    if content["startPageNumber"] == content["endPageNumber"]:
        if any(page.get("pageNumber") != number for page in content.get("pages", [])):
            raise ValueError("Page metadata contradicts the single-page content range")
        return markdown
    if encoding != "codePoint":
        raise ValueError("Multi-page slicing requires codePoint offsets")
    pages = [page for page in content.get("pages", []) if page.get("pageNumber") == number]
    if len(pages) != 1 or not pages[0].get("spans"):
        raise ValueError("Cannot isolate this page: pages[].spans are missing or ambiguous")
    fragments: list[str] = []
    previous_end = -1
    for span in pages[0]["spans"]:
        offset, length = span.get("offset", 0), span["length"]
        if offset < previous_end or offset + length > len(markdown):
            raise ValueError("Page spans must be ordered and within the original Markdown")
        fragments.append(markdown[offset:offset + length])
        previous_end = offset + length
    return "\n".join(fragments)


def prepare_pages(
    raw: JsonObject,
    analyzer_definition: JsonObject,
    *,
    complete_document: bool = False,
    document_level_fields: tuple[str, ...] = DOCUMENT_LEVEL_FIELDS,
    max_context_characters: int = 1000000,
    record_field: str | None = None,
    config: EvaluationConfig | None = None,
) -> list[JsonObject]:
    """Prepare every page before inference using the supplied CU analyzer definition.

    Args:
        raw: Successful raw CU output, optionally wrapped in result.
        analyzer_definition: Original analyzer definition or its fieldSchema.
        complete_document: Caller assertion; only a sole page can cover document-wide rules.
        document_level_fields: Analyzer paths requiring complete-document evidence.
        max_context_characters: Hard per-page evidence budget; content is never truncated.
        record_field: Optional single top-level record array; configured table_row paths can select multiple arrays.
        config: Optional shared offline/online field-model and canonical table-path selectors.

    Returns:
        Physically ordered page contexts with schema errors, original indices, and deferrals.

    Raises:
        TypeError: An input has an unsupported type.
        ValueError: Schema, status, page isolation, or evidence budget is invalid.
    """
    if not isinstance(complete_document, bool):
        raise TypeError("complete_document must be a Boolean caller assertion")
    if not isinstance(raw, dict):
        raise TypeError("cu_result must be a dictionary")
    if "status" in raw and str(raw["status"]).lower() != "succeeded":
        raise ValueError("CU analysis must have status Succeeded")
    schema, validator = analyzer_schema(analyzer_definition)
    definitions = schema["fields"]
    arrays = [name for name, node in definitions.items() if node["type"] == "array" and node["items"]["type"] == "object"]
    configured_rows: set[str] | None = None
    infer_tables = False
    if config is not None and any(rule.level == "table_row" for rule in config.tables.values()):
        response = parse_cu_result(raw).response
        configured_rows = {path for pattern, rule in config.tables.items() if rule.level == "table_row" for path, _value in slots(response, pattern) if config.selects_path(path, include_ancestors=True)}
        if any(len(path.split("/")) != 5 or path.split("/")[1] != "segments" or path.split("/")[3] != "fields" for path in configured_rows):
            raise ValueError("Page evaluation table_row paths must select a top-level analyzer record array")
        selected_arrays = {path.split("/")[4].replace("~1", "/").replace("~0", "~") for path in configured_rows}
        if not selected_arrays or selected_arrays - set(arrays):
            raise ValueError("Page evaluation table_row paths must select analyzer arrays of object records")
        if record_field is not None and selected_arrays != {record_field}:
            raise ValueError("record_field conflicts with the configured table_row paths")
        record_fields = [name for name in arrays if name in selected_arrays]
        infer_tables = not any(rule.level == "table_count" for rule in config.tables.values())
    else:
        record_field = record_field or ("lineItems" if "lineItems" in arrays else arrays[0] if len(arrays) == 1 else None)
        record_fields = [record_field] if record_field else []
    if any(name not in arrays for name in record_fields):
        raise ValueError("record_field must identify an analyzer array of object records")
    result = raw.get("result", raw)
    contents = result.get("contents")
    if not isinstance(contents, list) or not contents or not all(isinstance(content, dict) for content in contents):
        raise ValueError("CU result must contain document content blocks")
    locations = page_locations(contents)
    contexts: list[JsonObject] = []
    for content_index, number in locations:
        content = contents[content_index]
        start, end = content["startPageNumber"], content["endPageNumber"]
        single_page = start == end
        markdown = page_markdown(content, number, result.get("stringEncoding", raw.get("stringEncoding", "codePoint")))
        if len(markdown) > max_context_characters:
            raise ValueError("Page exceeds configured context budget; never truncate evidence")
        fields = content.get("fields", {})
        normalized = cu_value({"type": "object", "valueObject": fields})
        errors = [{"path": list(error.absolute_path), "message": error.message} for error in validator.iter_errors(normalized)]
        targets = collect_targets(fields, definitions)
        if config is not None:
            targets = [target for target in targets if config.selects_path(f"/segments/{content_index}/fields{target['pointer']}", include_ancestors=True)]
        records: list[JsonObject] = []
        deferred_records: list[JsonObject] = []
        record_pages: dict[str, set[int]] = {}
        table_groups: list[list[int]] = []
        array_presence: list[bool] = []
        record_offset = 0
        for name in record_fields:
            if configured_rows is not None and pointer(f"/segments/{content_index}/fields", name) not in configured_rows:
                continue
            array = fields.get(name, {})
            present = array.get("type") == "array" and isinstance(array.get("valueArray"), list)
            array_presence.append(present)
            items = array["valueArray"] if present else []
            group: list[int] = []
            for index, item in enumerate(items):
                path = f"{name}[{index}]"
                numbers = source_pages(item) or ({number} if single_page else set())
                record_pages[path] = numbers
                if any(page < start or page > end for page in numbers):
                    deferred_records.append({"path": path, "source_pages": sorted(numbers), "reason": "Record source is outside the content page range."})
                elif not numbers or len(numbers) > 1:
                    if not numbers or number in numbers:
                        deferred_records.append({"path": path, "source_pages": sorted(numbers), "reason": "Missing or cross-page record provenance."})
                elif number in numbers:
                    records.append({"path": path, "index": record_offset + index, "values": cu_value(item), "source_pages": sorted(numbers)})
                    group.append(record_offset + index)
            record_offset += len(items)
            if present and (single_page or group or number in source_pages(array)):
                table_groups.append(group)
        local: list[JsonObject] = []
        deferred: list[JsonObject] = []
        other: list[str] = []
        whole = complete_document and len(locations) == 1
        for target in targets:
            path, field = target["path"], target["cu_field"]
            numbers = source_pages(field)
            match = re.match(r"^(.+\[[0-9]+\])\.", path)
            row_pages = record_pages.get(match.group(1), set()) if match else set()
            if not numbers and len(row_pages) == 1:
                numbers = set(row_pages)
            if not numbers and single_page:
                numbers = {number}
            if field and field.get("type") != target["field_schema"]["type"]:
                errors.append({"path": path, "message": "CU wrapper type disagrees with analyzer schema."})
            reason = None
            if any(page < start or page > end for page in numbers):
                reason = "Field source is outside the content page range."
            elif numbers and number not in numbers:
                other.append(path)
                continue
            elif not numbers or len(numbers) > 1:
                reason = "Missing or cross-page field provenance."
            elif match and (len(row_pages) != 1 or any(page < start or page > end for page in row_pages)):
                reason = "Shipment record has cross-page or unresolved fields."
            elif path in document_level_fields and not whole:
                reason = "Schema requires document-level evidence."
            if reason:
                deferred.append({"path": path, "reason": reason})
            else:
                local.append({key: target[key] for key in ("path", "field_schema", "parent_schema", "value")} | {"source_pages": sorted(numbers)})
        contexts.append({
            "content_index": content_index, "page_number": number, "page_key": f"content_{content_index}_page_{number}",
            "markdown": markdown, "is_complete_document": whole, "fields": local, "records": records,
            "deferred_records": deferred_records, "record_array_present": bool(array_presence) and all(array_presence),
            "deferred_fields": deferred, "other_page_fields": other, "content_structural_errors": errors,
            "cu_warnings": result.get("warnings", []), "shipment_schema": definitions[record_fields[0]] if len(record_fields) == 1 else {"type": "object", "properties": {name: definitions[name] for name in record_fields}} if record_fields else {},
            "analyzer_schema": schema, "document_level_fields": list(document_level_fields),
            **({"configured_table_groups": table_groups} if infer_tables else {}),
        })
    return contexts