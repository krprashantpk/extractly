"""Demonstrate reference-free Python evaluation with caller-owned authentication."""

import json
from pathlib import Path

from azure.core.credentials import TokenCredential

from extractly import OnlineEvaluationConfig, OnlineEvaluator
from extractly.types import JsonObject


def evaluate_online_sample(*, credential: TokenCredential | None = None) -> JsonObject:
    """Return an online report using explicitly configured Azure semantic graders.

    Args:
        credential: Caller-configured credential, retained and closed only by the caller.

    Returns:
        Reference-free evaluation report for the synthetic document.

    Raises:
        ValueError: Configuration, input, or evidence policy is invalid.
        RuntimeError: A service result violates the expected contract.

    Replace model placeholders in the online JSON before calling. This function
    sends source evidence to configured Azure services and may incur charges.
    Provider exceptions propagate; the function does not write a report file.
    """
    sample = Path(__file__).parent
    config = OnlineEvaluationConfig.model_validate_json((sample / "online.json").read_text(encoding="utf-8"))
    document = json.loads((sample / "synthetic.jsonl").read_text(encoding="utf-8"))
    return OnlineEvaluator(config, credential=credential).evaluate(document["cu_result"], document_id=document["id"])
