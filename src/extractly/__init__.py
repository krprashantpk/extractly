"""Evaluate Content Understanding extraction with Microsoft Foundry."""

from .comparison import EvaluationModel, field_comparator
from .config import EvaluationConfig, TableRule
from .cu import ParsedCUResult, ParsedCUSegment, parse_cu_result
from .eval_runner import EvaluationRunner, FoundryEvaluationRunner, RunnerRequest, RunnerResult
from .evaluators import FieldEvaluator, LLMFieldEvaluator, LLMStructuralEvaluator, SchemaEvaluator, StructuralEvaluator
from .offline_pipeline import OfflineEvaluationConfig, OfflineEvaluationPipeline
from .online_pipeline import OnlineEvaluationConfig, OnlineEvaluationPipeline
from .store import AzureBlobGoldenDatasetStore, GoldenDatasetStore

__version__ = "0.1.0"

__all__ = [
    "AzureBlobGoldenDatasetStore",
    "EvaluationConfig",
    "EvaluationModel",
    "EvaluationRunner",
    "FieldEvaluator",
    "FoundryEvaluationRunner",
    "GoldenDatasetStore",
    "LLMFieldEvaluator",
    "LLMStructuralEvaluator",
    "OfflineEvaluationConfig",
    "OfflineEvaluationPipeline",
    "OnlineEvaluationConfig",
    "OnlineEvaluationPipeline",
    "ParsedCUResult",
    "ParsedCUSegment",
    "RunnerRequest",
    "RunnerResult",
    "SchemaEvaluator",
    "StructuralEvaluator",
    "TableRule",
    "__version__",
    "field_comparator",
    "parse_cu_result",
]
