"""Coordinate independent online checks and retain physical-page execution progress."""

import json
from dataclasses import dataclass

from typesafe_sdk import Choice

from ..config import EvaluationConfig
from ..cu._pages import DOCUMENT_LEVEL_FIELDS, prepare_pages
from ..types import JsonObject
from ._decision import DecisionRequest, choice_result, judge
from ._shared import ratios
from .online_field import OnlineFieldEvaluator
from .online_structural import PAGE_STRUCUTURAL_INTEGRATY, OnlineStructuralEvaluator


@dataclass(frozen=True)
class PageRequest:
	"""Keep page preparation and independent questions separate from inference."""

	context: JsonObject
	field: DecisionRequest
	structural: DecisionRequest

	@property
	def state(self) -> JsonObject:
		"""Return shared page-only evidence without internal scoring metadata."""
		return {key: value for key, value in (self.field.state | self.structural.state).items() if key != "structural_prechecks"}

	@property
	def questions(self) -> dict[str, Choice]:
		"""Combine independent checks into exactly one request per eligible page."""
		return self.field.questions | self.structural.questions

	def preview(self) -> JsonObject:
		"""Expose the exact JSON request and local prechecks before service calls."""
		return {"page_key": self.context["page_key"], "state": self.state, "questions": {name: question.model_dump(exclude_none=True) for name, question in self.questions.items()}, "structural_prechecks": self.structural.state["structural_prechecks"]}


class OnlineChecks:
	"""Combine prepared checks into one request with caller-owned client execution."""

	def __init__(self, config: EvaluationConfig, field: OnlineFieldEvaluator | None, structural: OnlineStructuralEvaluator | None) -> None:
		"""Require at least one check and one shared client when both are supplied."""
		if field is not None and not isinstance(field, OnlineFieldEvaluator):
			raise TypeError("field_evaluator must be an OnlineFieldEvaluator or None")
		if structural is not None and not isinstance(structural, OnlineStructuralEvaluator):
			raise TypeError("structural_evaluator must be an OnlineStructuralEvaluator or None")
		clients = [evaluator.client for evaluator in (field, structural) if evaluator is not None]
		if not clients:
			raise ValueError("At least one online check is required")
		if any(client is not clients[0] for client in clients):
			raise ValueError("Field and structural evaluators must share the same caller-owned client")
		self.config = config
		self.field = field
		self.structural = structural
		self.client = clients[0]

	def configuration(self) -> JsonObject:
		"""Describe reproducible scoring policy without serializing client secrets."""
		return {
			"policy_version": 2,
			"field_min_probability": self.field.min_probability if self.field else None,
			"structural_min_probability": self.structural.min_probability if self.structural else None,
			"source_count_limits": self.structural.count_limits if self.structural else None,
		}

	def __call__(self, response: str, context: str, contract: str, field_rows: list[JsonObject] | None = None, cu_warnings: list[JsonObject] | None = None) -> JsonObject:
		"""Prepare both checks before one service call and retain incomplete metric scope.

		Args:
			response: Canonical response prepared by CUAdapter.
			context: Full document Markdown within the explicit context budget.
			contract: Serialized model and table policy, with no reference labels.
			field_rows: Selected field slots prepared by the existing adapter.
			cu_warnings: Provider warnings that independently require review.

		Returns:
			Field and structural metrics plus judgments; full field scores require model-selected slots and established row coverage.

		Provider exceptions propagate unchanged. The caller owns client retries and cleanup.
		"""
		empty = DecisionRequest({}, {})
		field_request = self.field.prepare(field_rows or [], context, contract) if self.field else empty
		structural_request = self.structural.prepare(response, context, contract) if self.structural else empty
		answers = judge(self.client, {**field_request.state, **structural_request.state}, {**field_request.questions, **structural_request.questions}, min_probability=0.0)
		fields = self.field.score(field_request, answers) if self.field else {}
		structure = self.structural.score(structural_request, answers) if self.structural else {}
		complete = bool(self.field and self.config.extraction_schema is not None and structure.get("rows_complete"))
		metrics = {name: fields.get(name) if complete else None for name in ("tp", "fp", "fn", "precision", "recall", "f1")}
		metrics.update({name: structure.get(name) for level in ("table_row", "table_count") for name in (f"{level}_{key}" for key in ("tp", "fp", "fn", "precision", "recall", "f1"))})
		exact = [fields.get("exact_match"), structure.get("exact_match")]
		return {
			**metrics, "exact_match": 0 if 0 in exact else 1 if complete and exact == [1, 1] else None,
			"field_counts": fields.get("field_counts", {}), "field_judgments": fields.get("judgments", {}),
			"structural_checks": structure.get("checks", {}), "source_counts": structure.get("source_counts", {}),
			"review_required": bool(cu_warnings) or not complete or fields.get("review_required", False) or structure.get("review_required", False),
		}

	def prepare_pages(
		self,
		raw: JsonObject,
		analyzer_definition: JsonObject,
		*,
		complete_document: bool = False,
		table_groups: dict[str, list[list[int]]] | None = None,
		selected_field_paths: list[str] | None = None,
		document_level_fields: tuple[str, ...] = DOCUMENT_LEVEL_FIELDS,
		record_field: str | None = None,
		max_fields_per_document: int = 2000,
	) -> list[PageRequest]:
		"""Validate every page, field selection, grouping, and budget before inference."""
		pages = prepare_pages(raw, analyzer_definition, complete_document=complete_document, document_level_fields=document_level_fields, max_context_characters=self.config.max_context_characters, record_field=record_field, config=self.config)
		available = {target["path"] for page in pages for target in page["fields"]}
		if selected_field_paths is not None:
			if not isinstance(selected_field_paths, list) or not all(isinstance(path, str) for path in selected_field_paths):
				raise TypeError("selected_field_paths must be a list of strings or None")
			if len(selected_field_paths) != len(set(selected_field_paths)) or set(selected_field_paths) - available:
				raise ValueError("Selected paths must be unique and page-local on at least one prepared page; inspect deferred_fields")
		groups = {} if table_groups is None else table_groups
		if not isinstance(groups, dict):
			raise TypeError("table_groups must be a dictionary or None")
		if set(groups) - {page["page_key"] for page in pages}:
			raise ValueError("table_groups contains an unknown page_key")
		groups = dict(groups)
		for page in pages:
			if "configured_table_groups" in page:
				configured = page["configured_table_groups"]
				if page["page_key"] in groups and groups[page["page_key"]] != configured:
					raise ValueError("table_groups conflicts with configured table_row list containers")
				groups[page["page_key"]] = configured
		structure = self.structural or OnlineStructuralEvaluator(self.client)
		requests = [PageRequest(page, self.field.prepare_page(page, selected_field_paths=selected_field_paths) if self.field else DecisionRequest({"fields": {}}, {}), structure.prepare_page(page, table_groups=groups.get(page["page_key"]), include_structure=self.structural is not None)) for page in pages]
		if sum(len(request.field.state["fields"]) for request in requests) > max_fields_per_document:
			raise ValueError("Document exceeds configured field-grading budget")
		json.dumps([request.preview() for request in requests], allow_nan=False)
		return requests

	def page_report(self, request: PageRequest, answers: JsonObject, *, status: str) -> JsonObject:
		"""Assemble notebook-shaped field, structure, schema, and execution evidence."""
		context, state = request.context, request.state
		field_result = self.field.score(request.field, answers) if self.field else {"field_counts": {}}
		threshold = self.field.min_probability if self.field else self.structural.min_probability if self.structural else 0.8
		fields: list[JsonObject] = []
		for path, candidate in state["fields"].items():
			result = choice_result(answers, path, threshold)
			presence_name = f"SOURCE_PRESENCE.{path}"
			presence = choice_result(answers, presence_name if presence_name in request.field.questions else None, threshold)
			counts = field_result["field_counts"][path]
			fields.append({"path": path, "value": candidate["value"], **result, "expected_presence": presence, "occurrence_counts": counts, "review_required": counts is None or result["verdict"] not in ("tp", "tn")})
		structure = (self.structural or OnlineStructuralEvaluator(self.client)).score_page(request.structural, answers)
		checks = structure["checks"]
		selected_all = self.field is not None and set(state["fields"]) == {target["path"] for target in context["fields"]}
		rows_covered = all(checks[name]["supported"] and checks[name]["verdict"] == "pass" for name in ("missing_rows", "extra_rows", "duplicate_rows"))
		scope_complete = selected_all and not context["deferred_fields"] and not context["deferred_records"] and rows_covered
		complete = scope_complete and all(field["occurrence_counts"] is not None for field in fields)
		observed = [sum(field["occurrence_counts"][index] for field in fields if field["occurrence_counts"] is not None) for index in range(3)]
		assessed = selected_all and not context["deferred_fields"] and not context["deferred_records"] and status == "completed"
		metrics = {f"extraction.{name}": observed[index] if assessed else None for index, name in enumerate(("tp", "fp", "fn"))}
		metrics.update({f"extraction.{name}": value for name, value in (ratios(*observed) if assessed else dict.fromkeys(("precision", "recall", "f1"))).items()})
		metrics.update({f"extraction.{name}": value for name, value in structure.items() if name.startswith(("table_row_", "table_count_"))})
		return {
			"page_key": context["page_key"], "content_index": context["content_index"], "page_number": context["page_number"],
			"evaluation_status": status, "complete_document_asserted": context["is_complete_document"],
			"schema_validation_scope": "content_block", "content_structural_errors": context["content_structural_errors"], "cu_warnings": context["cu_warnings"],
			"fields": fields, "field_metrics_complete": complete, "assessed_field_counts": dict(zip(("tp", "fp", "fn"), observed, strict=True)),
			PAGE_STRUCUTURAL_INTEGRATY: {"scope": "page", **{key: state[key] for key in ("extracted_row_count", "extracted_table_count", "table_group_errors")}, "checks": checks, "source_counts": structure["source_counts"], "review_required": structure["review_required"]},
			"metrics": metrics, "deferred_records": context["deferred_records"], "deferred_fields": context["deferred_fields"],
			"other_page_fields_not_evaluated_here": context["other_page_fields"], "review_probability_threshold": threshold,
			"review_required": status != "completed" or not complete or structure["review_required"] or any(field["review_required"] for field in fields) or bool(context["content_structural_errors"]) or bool(context["cu_warnings"]),
		}

	def evaluate_pages(self, requests: list[PageRequest], *, progress: JsonObject) -> JsonObject:
		"""Run sequentially and retain validated earlier pages if a later request raises.

		Args:
			requests: Every page prepared before the first service operation.
			progress: Caller-owned destination containing page reports and execution status.

		Returns:
			Document metrics and detailed page coverage; fragments never imply document completeness.

		Provider exceptions propagate unchanged; retries, clients, and cleanup remain caller-owned.
		"""
		progress.update(pages=[self.page_report(request, {}, status="not_run" if request.questions else "not_evaluated") for request in requests])
		progress.update(pages_prepared=len(requests), pages_completed=0, evaluation_status="not_run", document_level_evaluation_status="covered_by_single_page" if len(requests) == 1 and requests[0].context["is_complete_document"] else "not_evaluated", document_level_fields=list(requests[0].context["document_level_fields"]) if requests else [], review_required=True)
		for index, request in enumerate(requests):
			if not request.questions:
				continue
			progress["pages"][index] = self.page_report(request, {}, status="failed")
			progress["evaluation_status"] = "partial" if progress["pages_completed"] else "failed"
			metadata: JsonObject = {}
			answers = judge(self.client, request.state, request.questions, min_probability=0.0, metadata=metadata)
			progress["pages"][index] = self.page_report(request, answers, status="completed") | metadata
			progress["pages_completed"] += 1
		pages = progress["pages"]
		whole = len(pages) == 1 and pages[0]["complete_document_asserted"]
		metrics: JsonObject = {}
		for level in ("", "table_row_", "table_count_"):
			counts = [[page["metrics"][f"extraction.{level}{name}"] for name in ("tp", "fp", "fn")] for page in pages]
			known = (whole or level == "table_row_") and all(value is not None for count in counts for value in count)
			totals = [sum(count[index] for count in counts) for index in range(3)] if known else None
			values = dict.fromkeys(("tp", "fp", "fn", "precision", "recall", "f1")) if totals is None else dict(zip(("tp", "fp", "fn"), totals, strict=True)) | ratios(*totals)
			metrics.update({f"{level}{name}": value for name, value in values.items()})
		errors = {page["content_index"]: page["content_structural_errors"] for page in pages}
		violations = sum(len(items) for items in errors.values())
		known_error = violations > 0 or any(field["supported"] and field["verdict"] in ("fp", "fn") for page in pages for field in page["fields"]) or any((check["supported"] or check["origin"] == "local_precheck") and check["verdict"] == "fail" for page in pages for check in page[PAGE_STRUCUTURAL_INTEGRATY]["checks"].values())
		review = not whole or any(page["review_required"] for page in pages)
		completed = sum(page["evaluation_status"] == "completed" for page in pages)
		progress.update(pages_prepared=len(pages), pages_completed=completed, evaluation_status="completed" if completed == len(pages) else "partial" if completed else "not_run", document_level_evaluation_status="covered_by_single_page" if whole else "not_evaluated", review_required=review)
		return {**metrics, "exact_match": 0 if known_error else 1 if not review and metrics["tp"] is not None else None, "review_required": review, "schema_valid": int(not violations), "schema_violations": violations}