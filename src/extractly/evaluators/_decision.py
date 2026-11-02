"""Validate Decision choices shared by reference-free field and structure checks."""

import json
import logging
from dataclasses import dataclass

from typesafe_sdk import Choice, TypeSafeClient

from ..types import JsonObject

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionRequest:
    """Keep JSON evidence and named questions separate from client execution."""

    state: JsonObject
    questions: dict[str, Choice]


def supported_answers(request: DecisionRequest, answers: JsonObject, min_probability: float) -> JsonObject:
    """Apply one evaluator's threshold to its own validated answers."""
    return {name: choice_result(answers, name, min_probability) for name in request.questions}


def choice_result(answers: JsonObject, name: str | None, min_probability: float) -> JsonObject:
    """Keep unasked and unrun choices explicit instead of fabricating successful answers."""
    result = answers.get(name) if name is not None else None
    if result is None:
        return {"verdict": "not_evaluated" if name is None else "not_run", "probability": None, "confidence": None, "probabilities": None, "supported": False}
    return {**result, "supported": result["probability"] >= min_probability}


def validate_judge(client: TypeSafeClient, min_probability: float) -> None:
    """Validate caller-owned Decision dependencies and the review threshold."""
    if not callable(getattr(client, "system_one", None)):
        raise TypeError("client must provide a callable system_one method")
    if isinstance(min_probability, bool) or not isinstance(min_probability, (int, float)):
        raise TypeError("min_probability must be a number, not a boolean")
    if not 0 <= min_probability <= 1:
        raise ValueError("min_probability must be between zero and one")


def judge(client: TypeSafeClient, state: JsonObject, questions: dict[str, Choice], *, min_probability: float, metadata: JsonObject | None = None) -> JsonObject:
    """Return validated Choice judgments without configuring or closing the client.

    Args:
        client: Caller-configured synchronous Decision client.
        state: JSON evidence and candidate values; never reference labels.
        questions: Named Choice questions sharing the supplied state.
        min_probability: Selected-label probability required to support a judgment.
        metadata: Optional caller-owned destination for validated response model and usage metadata.

    Returns:
        Judgments retaining verdict, probabilities, confidence, and support status; empty questions make no call.

    Raises:
        RuntimeError: Decision returns missing answers, unexpected labels, or invalid probabilities.
        ValueError: State is not finite, serializable JSON.

    Provider exceptions propagate unchanged. Probabilities are not calibrated accuracy guarantees.
    """
    json.dumps(state, allow_nan=False)
    if not questions:
        return {}
    logger.info("Starting Decision evaluation of %d questions", len(questions))
    response = client.system_one(state=state, questions=questions)
    if set(response.answers) != set(questions) or set(response.choices) != set(questions):
        raise RuntimeError("Decision must return exactly one Choice answer per requested question")
    results: JsonObject = {}
    for name, answer in response.choices.items():
        labels = questions[name].criteria
        if answer.choice not in labels or set(answer.probabilities) != set(labels):
            raise RuntimeError(f"Decision returned unexpected labels or incomplete probabilities for '{name}'")
        values = [*answer.probabilities.values(), answer.confidence]
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in values):
            raise RuntimeError(f"Decision returned invalid probabilities or confidence for '{name}'")
        if abs(sum(answer.probabilities.values()) - 1) > 0.02:
            raise RuntimeError(f"Decision probabilities must sum to one for '{name}'")
        probability = answer.probabilities[answer.choice]
        results[name] = {
            "verdict": answer.choice, "probability": probability,
            "confidence": answer.confidence, "probabilities": dict(answer.probabilities),
            "supported": probability >= min_probability,
        }
    logger.info("Completed Decision evaluation of %d questions", len(results))
    if metadata is not None:
        usage = getattr(response, "usage", None)
        dump_usage = getattr(usage, "model_dump", None)
        metadata.update(model=getattr(response, "model", None), usage=dump_usage() if callable(dump_usage) else usage)
    return results