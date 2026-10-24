"""Shared boundary types for open-ended CU JSON and Foundry evaluator results.

Any is limited to JSON payloads and provider-owned result dictionaries whose
schemas are configured by the caller rather than fixed by this package.
"""

from collections.abc import Callable
from typing import Any

from azure.ai.evaluation import AzureOpenAIGrader

type JsonObject = dict[str, Any]
type Evaluator = Callable[..., JsonObject]
type EvaluationRunner = Callable[..., JsonObject]
type FoundryEvaluator = Callable[..., Any] | AzureOpenAIGrader
