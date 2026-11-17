"""Select extraction rules independently for each CU content item."""

from collections.abc import Sequence
from dataclasses import dataclass, replace

from ..types import JsonObject
from .cu_result_parser import ParsedCUSegment, parse_cu_result


@dataclass(frozen=True)
class AnalyzerSelection:
    """Pair detached content with caller-owned rules or an explicit skip outcome.

    Attributes:
        content: Parsed content with local Markdown and normalized fields.
        analyzer_definition: Selected caller-owned definition, or None when skipped or no definitions were supplied.
        skip_reason: Explicit reason for skipping content, or None when evaluation should proceed.

    Selected definitions are borrowed, not copied or modified.
    """

    content: ParsedCUSegment
    analyzer_definition: JsonObject | None
    skip_reason: str | None = None


def select_analyzer_definitions(analyzer_definitions: Sequence[JsonObject], raw: JsonObject) -> list[AnalyzerSelection]:
    """Select one definition per parsed content item without evaluating or saving it.

    Args:
        analyzer_definitions: Analyzer definitions; an empty sequence disables routing for custom evaluators.
        raw: One successful CU result, optionally wrapped in result.

    Returns:
        Ordered content selections, including explicit skips for unknown or unrouted classifier categories and missing referenced analyzers. A classifier is identified by config.contentCategories, never by its name. Without a classifier, one definition applies to all content; multiple definitions require a matching content or result analyzerId.

    Raises:
        TypeError: Definitions are not a sequence of dictionaries or raw is not a dictionary.
        ValueError: Definitions, CU content, or routing identities are invalid or ambiguous.

    Caller-owned inputs are unchanged. Classifier declarations can provide a
    routed child's category, but never its source Markdown or extracted fields.
    """
    if not isinstance(analyzer_definitions, Sequence) or isinstance(analyzer_definitions, (str, bytes)):
        raise TypeError("analyzer_definitions must be a sequence of dictionaries")
    if not isinstance(raw, dict):
        raise TypeError("raw must be one CU result dictionary")
    by_id, classifier = _index_definitions(analyzer_definitions)
    parsed = parse_cu_result(raw)
    payload = raw.get("result", raw)
    metadata, content_paths = _content_metadata(payload)
    selections: list[AnalyzerSelection] = []
    for content in parsed.segments:
        item = metadata[content.content_path]
        if content.category is None and item.get("category") is not None:
            content = replace(content, category=item["category"])
        if classifier is not None:
            selection = _select_classified(content, item, classifier, by_id)
            if selection.skip_reason is None and content.content_path not in content_paths:
                selection = AnalyzerSelection(content, None, "Classified segment has no routed extraction content")
            selections.append(selection)
        elif len(analyzer_definitions) <= 1:
            selections.append(_select_single(content, analyzer_definitions))
        else:
            analyzer_id = item.get("analyzerId", payload.get("analyzerId", raw.get("analyzerId")))
            selections.append(_select_by_id(content, analyzer_id, by_id))
    return selections


def _index_definitions(definitions: Sequence[JsonObject]) -> tuple[dict[str, JsonObject], JsonObject | None]:
    """Validate identities and classifier routes before processing any content."""
    by_id: dict[str, JsonObject] = {}
    classifier: JsonObject | None = None
    for definition in definitions:
        if not isinstance(definition, dict):
            raise TypeError("Every analyzer_definition must be a dictionary")
        analyzer_id = definition.get("analyzerId")
        if analyzer_id is not None or len(definitions) > 1:
            if not isinstance(analyzer_id, str) or not analyzer_id.strip():
                raise ValueError("analyzerId must be a nonempty string and is required for multiple analyzer_definitions")
            if analyzer_id in by_id:
                raise ValueError("analyzer_definitions must have unique analyzerId values")
            by_id[analyzer_id] = definition
        config = definition.get("config", {})
        if not isinstance(config, dict):
            raise ValueError("Analyzer config must be a dictionary")
        if "contentCategories" not in config:
            continue
        if classifier is not None:
            raise ValueError("analyzer_definitions must contain at most one classifier")
        categories = config["contentCategories"]
        if not isinstance(categories, dict) or not categories:
            raise ValueError("Classifier contentCategories must be a nonempty dictionary")
        for category, route in categories.items():
            if not isinstance(category, str) or not category.strip() or not isinstance(route, dict):
                raise ValueError("Classifier categories require nonempty names and dictionary routes")
            target = route.get("analyzerId")
            if target is not None and (not isinstance(target, str) or not target.strip()):
                raise ValueError("Classifier route analyzerId must be a nonempty string or None")
        classifier = definition
    return by_id, classifier


def _content_metadata(payload: JsonObject) -> tuple[dict[str, JsonObject], set[str]]:
    """Index content and inherit category metadata from classifier declarations."""
    metadata: dict[str, JsonObject] = {}
    for index, item in enumerate(payload["contents"]):
        path = item.get("path", f"content-{index}")
        for declaration in item.get("segments") or []:
            child_path = f"{path}/{declaration['segmentId']}"
            if child_path in metadata:
                raise ValueError("Duplicate declared CU segment path")
            metadata[child_path] = declaration
    content_paths: set[str] = set()
    for index, item in enumerate(payload["contents"]):
        path = item.get("path", f"content-{index}")
        content_paths.add(path)
        declaration = metadata.get(path, {})
        metadata[path] = {**declaration, **item}
        if item.get("category") is None and declaration.get("category") is not None:
            metadata[path]["category"] = declaration["category"]
    return metadata, content_paths


def _select_single(content: ParsedCUSegment, definitions: Sequence[JsonObject]) -> AnalyzerSelection:
    """Apply the sole definition, or leave custom evaluators unconfigured."""
    return AnalyzerSelection(content, definitions[0] if definitions else None)


def _select_classified(content: ParsedCUSegment, metadata: JsonObject, classifier: JsonObject, by_id: dict[str, JsonObject]) -> AnalyzerSelection:
    """Resolve the category's extraction analyzer or retain an explicit skip."""
    category = metadata.get("category")
    if category is not None and (not isinstance(category, str) or not category.strip()):
        raise ValueError("CU content category must be a nonempty string or None")
    route = classifier["config"]["contentCategories"].get(category)
    if route is None:
        return AnalyzerSelection(content, None, "Content category has no classifier route")
    analyzer_id = route.get("analyzerId")
    if analyzer_id is None:
        return AnalyzerSelection(content, None, "Classifier category has no extraction analyzerId")
    definition = by_id.get(analyzer_id)
    if definition is None:
        return AnalyzerSelection(content, None, "Classifier references an analyzer definition that was not supplied")
    if definition is classifier:
        raise ValueError("Classifier routes must reference an extraction analyzer, not the classifier")
    return AnalyzerSelection(content, definition)


def _select_by_id(content: ParsedCUSegment, analyzer_id: object, by_id: dict[str, JsonObject]) -> AnalyzerSelection:
    """Require an explicit matching identity rather than guessing by list order."""
    if not isinstance(analyzer_id, str) or not analyzer_id.strip() or analyzer_id not in by_id:
        raise ValueError("Multiple analyzer_definitions without a classifier require a matching content or result analyzerId")
    return AnalyzerSelection(content, by_id[analyzer_id])
