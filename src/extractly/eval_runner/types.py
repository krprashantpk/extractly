"""Define typed job requests and evaluator-derived execution results."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from ..types import Evaluator, JsonObject


@dataclass(frozen=True)
class RunnerRequest:
    """Describe one evaluation job without owning its dataset or evaluators.

    Attributes:
        data: Existing UTF-8 JSONL dataset path, valid for the duration of run.
        evaluators: Caller-owned named evaluator callables or grader objects.
        evaluation_name: Name identifying the job.
        azure_ai_project: Optional Foundry publishing destination.
        fail_on_evaluator_errors: Whether evaluator failures abort SDK execution.
    """

    data: Path
    evaluators: Mapping[str, Evaluator]
    evaluation_name: str
    azure_ai_project: str | None = None
    fail_on_evaluator_errors: bool = True


@dataclass(frozen=True)
class RunnerResult:
    """Return row outputs, aggregate metrics, and optional provider metadata.

    Attributes:
        rows: One row per dataset ID with inputs.id and outputs.name.metric keys.
        metrics: Aggregated metrics named evaluator.metric.
        metadata: Additional native result fields retained in the final report; rows and metrics are reserved and cannot override the typed fields.
    """

    rows: list[JsonObject]
    metrics: JsonObject
    metadata: JsonObject = field(default_factory=dict)