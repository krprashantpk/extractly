"""Evaluate the synthetic sample without model calls or credentials."""

import json
import logging
from pathlib import Path

from extractly import OfflineEvaluationConfig, OfflineEvaluator
from extractly.types import JsonObject

logger = logging.getLogger(__name__)


def evaluate_offline_sample() -> JsonObject:
    """Return an offline report using independently specified synthetic gold."""
    sample = Path(__file__).parent
    config = OfflineEvaluationConfig.model_validate_json((sample / "offline.json").read_text(encoding="utf-8"))
    document = json.loads((sample / "synthetic.jsonl").read_text(encoding="utf-8"))
    report = OfflineEvaluator(config).evaluate(document["cu_result"], document_id=document["id"], ground_truth=document["ground_truth"])
    logger.info("Evaluated %d documents with gate status '%s'", report["documents"], report["gate_status"])
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    evaluate_offline_sample()
