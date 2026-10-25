"""Provide synthetic CU fixtures and a controlled typed evaluation runner."""

import json
from datetime import date
from decimal import Decimal

import pytest

from extractly import (
    Comparison,
    CUAdapter,
    EvaluationConfig,
    EvaluationModel,
    OfflineEvaluationConfig,
    OnlineEvaluationConfig,
    TableRule,
    field_comparator,
)
from extractly.eval_runner import EvaluationRunner, RunnerRequest, RunnerResult
from extractly.types import JsonObject


class Event(EvaluationModel):
    """Declare scored event values and an explicit raw-quantity tolerance."""

    eventId: str
    quantity: Decimal
    unit: str
    date: date

    @staticmethod
    @field_comparator("quantity")
    def within_ten_kilograms(expected: Decimal, actual: Decimal) -> Comparison:
        """Compare quantities in the source's declared kilogram unit."""
        return Comparison(expected == actual, abs(expected - actual) <= Decimal("10"))


class Events(EvaluationModel):
    """Select typed event rows within each canonical segment."""

    items: list[Event]


@pytest.fixture
def config() -> EvaluationConfig:
    """Return explicit row identity and model-owned comparison policy."""
    return EvaluationConfig(extraction_schema=Events, tables={"/segments": TableRule(keys=["category", "pages"], level="instance"), "/segments/*/fields/items": TableRule(keys=["eventId"])})


@pytest.fixture
def quantity_schema() -> type[EvaluationModel]:
    """Select quantities alone for grading and workflow selection checks."""
    class Quantity(EvaluationModel):
        """Select a typed quantity without the row's context fields."""

        quantity: Decimal

    class Items(EvaluationModel):
        """Apply quantity selection to every item occurrence."""

        items: list[Quantity]

    return Items


@pytest.fixture
def offline_config(config: EvaluationConfig) -> OfflineEvaluationConfig:
    """Configure offline scoring using the shared extraction contract."""
    return OfflineEvaluationConfig.model_validate(config.model_dump())


@pytest.fixture
def online_config(config: EvaluationConfig) -> OnlineEvaluationConfig:
    """Configure online checks using the shared extraction contract."""
    return OnlineEvaluationConfig.model_validate(config.model_dump())


@pytest.fixture
def raw() -> JsonObject:
    """Return two invented event rows, never customer documents."""
    return {
        "stringEncoding": "codePoint",
        "contents": [{
            "path": "input1", "category": "Events", "startPageNumber": 1, "endPageNumber": 1,
            "markdown": "Events\nA: 12000 kg, 2026-01-01\nB: 8000 kg, 2026-01-02",
            "fields": {"items": {"type": "array", "valueArray": [
                {"valueObject": {
                    "eventId": {"valueString": key}, "quantity": {"type": "number", "valueNumber": number},
                    "unit": {"valueString": "kg"}, "date": {"valueDate": day},
                }}
                for key, number, day in [("A", 12000, "2026-01-01"), ("B", 8000, "2026-01-02")]
            ]}},
        }],
    }


@pytest.fixture
def record(config: EvaluationConfig, raw: JsonObject) -> JsonObject:
    """Prepare the synthetic response used to isolate comparator behavior."""
    return CUAdapter(config).prepare(raw, "document-1")


class LocalRunner(EvaluationRunner):
    """Execute deterministic evaluator callables without model services."""

    def run(self, request: RunnerRequest) -> RunnerResult:
        """Return typed results while preserving the common runner contract."""
        assert request.fail_on_evaluator_errors is True
        records = [json.loads(line) for line in request.data.read_text(encoding="utf-8").splitlines()]
        rows = []
        numeric: dict[str, list[float]] = {}
        for item in records:
            row = {"inputs.id": item["id"], "inputs.ground_truth": item.get("ground_truth", "")}
            for name, evaluator in request.evaluators.items():
                assert callable(evaluator)
                for key, value in evaluator(**item).items():
                    row[f"outputs.{name}.{key}"] = value
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        numeric.setdefault(f"{name}.{key}", []).append(value)
            rows.append(row)
        return RunnerResult(rows=rows, metrics={key: sum(values) / len(values) for key, values in numeric.items()})


@pytest.fixture
def local_runner() -> EvaluationRunner:
    """Return a caller-owned typed runner for local deterministic tests."""
    return LocalRunner()
