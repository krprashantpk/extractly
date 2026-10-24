"""Provide synthetic CU fixtures and a controlled Foundry-compatible runner.

Any is used only for the SDK's open-ended keyword and result dictionaries.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from extractly import CUAdapter, EvaluationConfig, OfflineEvaluationConfig, OnlineEvaluationConfig
from extractly.types import EvaluationRunner, JsonObject


@pytest.fixture
def config() -> EvaluationConfig:
    """Return explicit row identity and typed quantity/date comparison policy."""
    return EvaluationConfig.model_validate({
        "tables": {
            "/segments": {"keys": ["category", "pages"], "level": "instance"},
            "/segments/*/fields/items": {"keys": ["eventId"]},
        },
        "fields": {
            "/segments/*/fields/items/*/quantity": {
                "kind": "decimal", "required": True, "absolute_tolerance": "0.01",
                "unit_field": "unit", "unit_scales": {"kg": "0.001", "MT": "1"},
            },
            "/segments/*/fields/items/*/date": {"kind": "date", "required": True},
        },
    })


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


@pytest.fixture
def local_runner() -> EvaluationRunner:
    """Execute real deterministic callables behind the public runner contract."""
    def run(**kwargs: Any) -> JsonObject:
        """Return native-shaped results without model calls."""
        assert kwargs["fail_on_evaluator_errors"] is True
        records = [json.loads(line) for line in Path(kwargs["data"]).read_text(encoding="utf-8").splitlines()]
        rows = []
        numeric: dict[str, list[float]] = {}
        for item in records:
            row = {"inputs.id": item["id"], "inputs.ground_truth": item.get("ground_truth", "")}
            for name, evaluator in kwargs["evaluators"].items():
                for key, value in evaluator(**item).items():
                    row[f"outputs.{name}.{key}"] = value
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        numeric.setdefault(f"{name}.{key}", []).append(value)
            rows.append(row)
        return {"rows": rows, "metrics": {key: sum(values) / len(values) for key, values in numeric.items()}}

    return run
