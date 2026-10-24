"""Evaluate Content Understanding extraction with Microsoft Foundry."""

from .adapter import CUAdapter
from .config import EvaluationConfig, FieldRule, ModelSettings, TableRule
from .evaluators import ExtractionEvaluator, SchemaEvaluator
from .offline import OfflineEvaluationConfig, OfflineEvaluator
from .online import OnlineEvaluationConfig, OnlineEvaluator

__version__ = "0.1.0"

__all__ = [
    "CUAdapter",
    "EvaluationConfig",
    "ExtractionEvaluator",
    "FieldRule",
    "ModelSettings",
    "OfflineEvaluationConfig",
    "OfflineEvaluator",
    "OnlineEvaluationConfig",
    "OnlineEvaluator",
    "SchemaEvaluator",
    "TableRule",
    "__version__",
]
