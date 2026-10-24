"""Evaluate Content Understanding extraction with Microsoft Foundry."""

from .config import EvaluationConfig, FieldRule, ModelSettings, TableRule
from .cu import CUAdapter, build_ground_truth
from .eval_runner import EvaluationRunner, FoundryEvaluationRunner, RunnerRequest, RunnerResult
from .evaluators import ExtractionEvaluator, SchemaEvaluator
from .offline import OfflineEvaluationConfig, OfflineEvaluator
from .online import OnlineEvaluationConfig, OnlineEvaluator

__version__ = "0.1.0"

__all__ = [
    "CUAdapter",
    "EvaluationConfig",
    "EvaluationRunner",
    "ExtractionEvaluator",
    "FieldRule",
    "FoundryEvaluationRunner",
    "ModelSettings",
    "OfflineEvaluationConfig",
    "OfflineEvaluator",
    "OnlineEvaluationConfig",
    "OnlineEvaluator",
    "RunnerRequest",
    "RunnerResult",
    "SchemaEvaluator",
    "TableRule",
    "__version__",
    "build_ground_truth",
]
