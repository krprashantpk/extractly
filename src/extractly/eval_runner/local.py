"""Execute synchronous evaluator callables locally without Foundry orchestration."""

import json
import logging
import math
from collections.abc import Callable
from inspect import Parameter, signature

from azure.ai.evaluation import AzureOpenAIGrader

from ..types import JsonObject
from .base import EvaluationRunner
from .types import RunnerRequest, RunnerResult

logger = logging.getLogger(__name__)


class LocalRunner(EvaluationRunner):
    """Run caller-owned synchronous evaluators sequentially in this process.

    Evaluators receive matching dataset columns, or all columns when accepting
    keyword arguments. Numeric outputs are averaged across successful rows;
    booleans, NaN, and nonnumeric outputs are excluded from aggregate metrics.
    No publishing or authentication is performed. Callers remain responsible
    for any service calls made by their supplied evaluator callables.
    """

    def run(self, request: RunnerRequest) -> RunnerResult:
        """Read a JSONL dataset and return prefixed inputs, outputs, and means.

        Args:
            request: Caller-owned dataset and synchronous named evaluator callables.

        Returns:
            One row per record and numeric means named evaluator.metric.

        Raises:
            TypeError: The request, evaluator names, callables, or failure policy are invalid.
            ValueError: Publishing is requested or a dataset record is not an object.
            RuntimeError: An evaluator returns something other than a dictionary with string keys.
            OSError: The dataset cannot be read.

        Evaluator exceptions propagate unchanged when fail_on_evaluator_errors
        is true; otherwise failures are logged and their outputs are omitted.
        """
        if not isinstance(request, RunnerRequest):
            raise TypeError("request must be a RunnerRequest")
        if request.azure_ai_project is not None:
            raise ValueError("LocalRunner does not publish results; use FoundryEvaluationRunner for azure_ai_project")
        if not isinstance(request.fail_on_evaluator_errors, bool):
            raise TypeError("fail_on_evaluator_errors must be a bool")
        callables: dict[str, Callable[..., JsonObject]] = {}
        parameters: dict[str, set[str] | None] = {}
        for name, evaluator in request.evaluators.items():
            if not isinstance(name, str) or not name.strip():
                raise TypeError("Evaluator names must be nonempty strings")
            if isinstance(evaluator, AzureOpenAIGrader) or not callable(evaluator):
                raise TypeError("LocalRunner requires callable evaluators; use FoundryEvaluationRunner for AzureOpenAIGrader objects")
            callables[name] = evaluator
            arguments = signature(evaluator).parameters.values()
            parameters[name] = None if any(argument.kind is Parameter.VAR_KEYWORD for argument in arguments) else {argument.name for argument in arguments if argument.kind in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)}
        logger.info("Starting local evaluation '%s'", request.evaluation_name)
        rows: list[JsonObject] = []
        numeric: dict[str, list[float]] = {}
        with request.data.open(encoding="utf-8") as dataset:
            for line in dataset:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError("Every dataset record must be a JSON object")
                row = {f"inputs.{key}": value for key, value in item.items()}
                for name, evaluator in callables.items():
                    accepted = parameters[name]
                    arguments_by_name = item if accepted is None else {key: value for key, value in item.items() if key in accepted}
                    try:
                        output = evaluator(**arguments_by_name)
                    except Exception:
                        if request.fail_on_evaluator_errors:
                            raise
                        logger.exception("Evaluator '%s' failed for record '%s' in local evaluation '%s'", name, item.get("id"), request.evaluation_name)
                        continue
                    if not isinstance(output, dict) or any(not isinstance(key, str) for key in output):
                        raise RuntimeError(f"Evaluator '{name}' must return a dictionary with string keys")
                    for key, value in output.items():
                        row[f"outputs.{name}.{key}"] = value
                        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isnan(value):
                            numeric.setdefault(f"{name}.{key}", []).append(value)
                rows.append(row)
        metrics = {key: sum(values) / len(values) for key, values in numeric.items()}
        logger.info("Completed local evaluation '%s' of %d records", request.evaluation_name, len(rows))
        return RunnerResult(rows=rows, metrics=metrics)