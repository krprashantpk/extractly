"""Exercise the standalone field evaluator without credentials or network calls."""

import json
from collections.abc import Iterator
from copy import deepcopy

import httpx
import pytest
from openai import OpenAI
from pydantic import BaseModel, Field

from extractly.evaluators import LLMFieldEvaluator
from extractly.evaluators.llm_field import FIELD_PROMPT, FieldEvaluation, _document_metrics
from extractly.types import JsonObject


class Row(BaseModel):
    """Retain typed values and optional absences in each original row."""

    quantity: int
    active: bool
    note: str | None = None


class Extraction(BaseModel):
    """Select an aliased field and nested rows without unrelated candidate keys."""

    identity: str = Field(alias="document/id")
    rows: list[Row]
    missing: str = "not an observed value"


@pytest.fixture
def recorded_client() -> Iterator[tuple[OpenAI, list[JsonObject]]]:
    """Supply a caller-owned client with a recording local HTTP transport."""
    requests: list[JsonObject] = []

    def respond(request: httpx.Request) -> httpx.Response:
        """Record the request and return a minimal typed provider response."""
        requests.append(json.loads(request.content))
        content = {"fields": [], "summary": "Local transport response"}
        return httpx.Response(200, headers={"x-request-id": "local-request"}, json={"id": "local-completion", "object": "chat.completion", "created": 0, "model": "test", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(content)}}]})

    with OpenAI(api_key="local-test", base_url="https://example.test/v1/", http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
        yield client, requests


def test_given_four_inputs_when_called_then_model_scoped_payload_preserves_values(recorded_client: tuple[OpenAI, list[JsonObject]]) -> None:
    """Preserve zero, false, original data, missing defaults, and escaped aliases."""
    client, requests = recorded_client
    candidate = {"document/id": "D1", "rows": [{"quantity": 0, "active": False}], "unselected": 12}
    original = deepcopy(candidate)
    analyzer = {"analyzerId": "test", "fieldSchema": {"fields": {"rows": {"type": "array"}}}}

    result = LLMFieldEvaluator(client, "deployment")("Source D1", candidate, analyzer, Extraction)
    payload = json.loads(requests[0]["messages"][1]["content"])

    assert payload["markdown"] == "Source D1"
    assert payload["extracted_values"] == candidate == original
    assert payload["analyzer_definition"] == analyzer
    assert payload["extraction_schema"] == Extraction.model_json_schema(mode="validation")
    assert payload["field_targets"] == [{"path": "/document~1id", "value": "D1", "present": True}, {"path": "/rows/0/quantity", "value": 0, "present": True}, {"path": "/rows/0/active", "value": False, "present": True}, {"path": "/rows/0/note", "value": None, "present": False}, {"path": "/missing", "value": None, "present": False}]
    assert requests[0]["messages"][0]["content"] == FIELD_PROMPT
    assert set(result) == {"judgment", "metrics", "deployment", "latency_seconds", "usage"}
    assert result["judgment"] == {"fields": [], "summary": "Local transport response"}
    assert result["metrics"] == {"extraction.document.tp": 0, "extraction.document.tn": 0, "extraction.document.ambiguous": 0, "extraction.document.fp": 0, "extraction.document.fn": 0, "extraction.document.precision": None, "extraction.document.recall": None, "extraction.document.f1": None}
    assert "confidence" not in requests[0]["response_format"]["json_schema"]["schema"]["$defs"]["FieldAssessment"]["properties"]
    assert "needs_document_context" not in json.dumps(requests[0]["response_format"])
    assert "needs_document_context" not in FIELD_PROMPT
    assert not client.is_closed()


@pytest.mark.parametrize(("judgments", "counts", "scores"), [
    ([("tp", "present")] * 16 + [("tn", "absent")] * 16, [16, 16, 0, 0, 0], [1.0, 1.0, 1.0]),
    ([("tp", "present"), ("fp", "present"), ("fp", "absent"), ("fn", "present"), ("tn", "absent"), ("ambiguous", "ambiguous")], [1, 1, 1, 2, 2], [0.333, 0.333, 0.333]),
    ([("tn", "absent"), ("ambiguous", "ambiguous")], [0, 1, 1, 0, 0], [None, None, None]),
    ([("fp", "present")], [0, 0, 0, 1, 1], [0.0, 0.0, 0.0]),
    ([("fp", "absent")], [0, 0, 0, 1, 0], [0.0, None, 0.0]),
    ([("fn", "present")], [0, 0, 0, 0, 1], [None, 0.0, 0.0]),
    ([], [0, 0, 0, 0, 0], [None, None, None]),
])
def test_given_field_verdicts_when_metrics_calculated_then_counts_and_ratios_follow_offline_policy(judgments: list[tuple[str, str]], counts: list[int], scores: list[float | None]) -> None:
    """Keep TN and ambiguity separate while wrong present values contribute FP and FN."""
    evaluation = FieldEvaluation.model_validate({"fields": [{"path": f"/field/{index}", "verdict": verdict, "source_presence": presence, "extracted_value_json": "2" if verdict in ("tp", "fp") else None, "source_value_json": "2" if presence == "present" else None, "source_excerpt": None, "explanation": "Metric arithmetic fixture"} for index, (verdict, presence) in enumerate(judgments)], "summary": "Metric arithmetic fixture"})

    metrics = _document_metrics(evaluation)

    assert metrics == {f"extraction.document.{name}": value for name, value in zip(("tp", "tn", "ambiguous", "fp", "fn", "precision", "recall", "f1"), [*counts, *scores], strict=True)}


@pytest.mark.parametrize(("markdown", "values", "definition", "model", "exception"), [
    ("", {}, {"fields": {"name": {"type": "string"}}}, Extraction, ValueError),
    ("Source", [], {"fields": {"name": {"type": "string"}}}, Extraction, TypeError),
    ("Source", {}, {}, Extraction, ValueError),
    ("Source", {}, {"fields": {"name": {"type": "string"}}}, {}, TypeError),
])
def test_given_invalid_inputs_when_called_then_rejected_before_provider(markdown: str, values: JsonObject, definition: JsonObject, model: type[BaseModel], exception: type[Exception], recorded_client: tuple[OpenAI, list[JsonObject]]) -> None:
    """Reject invalid caller contracts without sending a provider request."""
    client, requests = recorded_client

    with pytest.raises(exception):
        LLMFieldEvaluator(client, "deployment")(markdown, values, definition, model)

    assert requests == []


def test_given_provider_refusal_when_called_then_failure_remains_visible() -> None:
    """Do not turn a provider refusal into an empty success-shaped judgment."""
    def refuse(request: httpx.Request) -> httpx.Response:
        """Return a refusal through the real SDK's response parser."""
        return httpx.Response(200, json={"id": "refused", "object": "chat.completion", "created": 0, "model": "test", "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": None, "refusal": "Cannot assess source"}}]})

    with OpenAI(api_key="local-test", base_url="https://example.test/v1/", http_client=httpx.Client(transport=httpx.MockTransport(refuse))) as client, pytest.raises(RuntimeError, match="refused field evaluation"):
        LLMFieldEvaluator(client, "deployment")("Source", {}, {"fields": {"name": {"type": "string"}}}, Extraction)