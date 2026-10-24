---
title: Document evaluation sample
description: Synthetic offline evaluation and explicitly configured semantic evaluation.
---

## Offline evaluation

From the repository root:

```powershell
uv sync --all-groups
uv run python samples\document-evaluation\offline.py
```

[The data](synthetic.jsonl) is synthetic and contains independently specified
canonical gold. [The offline policy](offline.json) requires perfect schema
conformance and at least 0.99 pooled extraction F1.

The [offline Python sample](offline.py) exposes `evaluate_offline_sample()`,
which calls `OfflineEvaluator.evaluate(...)` and returns the report dictionary.
No model, credential, or online configuration is involved.

```python
from extractly import OfflineEvaluationConfig, OfflineEvaluator

evaluator = OfflineEvaluator(OfflineEvaluationConfig())
report = evaluator.evaluate(raw_cu_result, document_id="document-1", ground_truth=gold)
batch_report = evaluator.run(
    [{"id": "document-1", "cu_result": raw_cu_result, "ground_truth": gold}]
)
```

## Online evaluation

Replace the placeholders in [the online policy](online.json) with a model endpoint
and deployment that support the evaluation services required by the Foundry SDK.
The preview API version is explicit sample configuration, not a guarantee that
every resource/model supports label graders.

Authenticate using your organization's approved Azure credential mechanism or
configure `model.api_key_env` to name an existing key environment variable.
Do not put keys into files.

```python
from extractly import OnlineEvaluationConfig, OnlineEvaluator

config = OnlineEvaluationConfig.model_validate_json(policy_json)
evaluator = OnlineEvaluator(config, credential=credential)
report = evaluator.evaluate(raw_cu_result, document_id="document-1")
batch_report = evaluator.run([{"id": "document-1", "cu_result": raw_cu_result}])
```

The [online Python sample](online.py) exposes
`evaluate_online_sample(credential=...)`. It does not create credentials or call
the offline evaluator.

Online evaluation ignores any `ground_truth` in batch input. It grades extraction
against Markdown source evidence. Groundedness, field labels, and completeness
are separate checks. The sample has no release thresholds; `not_configured`
does not indicate acceptance.

> [!IMPORTANT]
> This operation sends document evidence to Azure and may incur charges.
> Obtain data-owner approval before substituting real documents. Live model
> grading is not covered by the offline test suite.

## Resource ownership

Offline scoring needs no credentials. In your own application, pass
configured credentials to `OnlineEvaluator` and custom evaluators to either
workflow. The library does not close them. Input file loading, report persistence,
and process exit behavior belong to your application; no package CLI is installed.
