---
title: Extractly
description: Evidence-preserving evaluation for Azure Content Understanding extraction.
---

## Evaluate extraction, not text similarity

Extractly is a Python 3.12+ package for evaluating saved
Azure Content Understanding (CU) results. It preserves document evidence and
uses Microsoft Foundry to execute evaluators, aggregate metrics, and optionally
publish results.

* Compare extracted values with independent gold using `OfflineEvaluator`
* Check schema conformance without gold labels using `OnlineEvaluator`
* Enable model-based groundedness, five-label field assessment, and independent
  completeness assessment when you have approved Azure resources
* Measure field precision/recall/F1, exact match, and row/document-instance detection
* Preserve source spans, confidence, Markdown, missing fields, and duplicate rows
* Apply explicit release gates without treating model scores as calibrated risk

The package contains no application business rules, customer documents,
deployment configuration, database schemas, or automatic environment-file loading.
It is licensed under [MIT](LICENSE). It has not been published to PyPI.

## Install from source

Use [uv](https://docs.astral.sh/uv/) from the repository root:

```powershell
uv sync --all-groups
```

From another local project, add this repository's path:

```powershell
uv add C:\path\to\Extractly
```

## Offline Python API

`OfflineEvaluator` compares predictions against required independent gold.
It owns gold-based dataset preparation, typed comparison, and pooled metrics.
It does not instantiate online graders or accept model/credential settings.

```python
import json
from pathlib import Path

from extractly import OfflineEvaluationConfig, OfflineEvaluator

sample = Path("samples") / "document-evaluation"
config = OfflineEvaluationConfig.model_validate_json(
    (sample / "offline.json").read_text(encoding="utf-8")
)
document = json.loads((sample / "synthetic.jsonl").read_text(encoding="utf-8"))

evaluator = OfflineEvaluator(config)
report = evaluator.evaluate(
    document["cu_result"],
    document_id=document["id"],
    ground_truth=document["ground_truth"],
)
assert report["metrics"]["extraction.micro_f1"] == 1.0
```

For batches, every record requires `id`, `cu_result`, and `ground_truth`:

```python
report = evaluator.run([document])
```

## Online Python API

`OnlineEvaluator` assesses extraction without gold. It owns schema checks and
optional evidence-based groundedness, field-label, and completeness jobs.
It never runs offline comparison or text-F1 diagnostics.

For schema-only assessment, no model or credentials are needed:

```python
from extractly import OnlineEvaluationConfig, OnlineEvaluator

online = OnlineEvaluator(OnlineEvaluationConfig())
online_report = online.evaluate(document["cu_result"], document_id=document["id"])
```

For semantic checks, pass your application's configured credential:

```python
online_config = OnlineEvaluationConfig.model_validate_json(
    (sample / "online.json").read_text(encoding="utf-8")
)
semantic_evaluator = OnlineEvaluator(online_config, credential=credential)
online_report = semantic_evaluator.evaluate(
    document["cu_result"], document_id=document["id"]
)
```

Online batch records require only `id` and `cu_result`:

```python
online_report = online.run([{"id": document["id"], "cu_result": document["cu_result"]}])
```

Online input records may contain other application fields; any `ground_truth`
is ignored and never included in evaluator datasets. The single-document online
method has no `ground_truth` parameter.

Alternatively, `model.api_key_env` names an existing environment variable.
Keys never belong in configuration files. The package never creates or closes
credentials.

## Independent workflows, reusable components

Online and offline evaluation have separate modules, configuration types,
single-document methods, and batch orchestration. Neither calls the other.
`EvaluationConfig` holds only the common extraction contract used by `CUAdapter`,
`SchemaEvaluator`, and `ExtractionEvaluator`. Use the corresponding
`OfflineEvaluationConfig` or `OnlineEvaluationConfig` with the public workflow.

Both workflows reuse CU adaptation, schema validation, Foundry execution/coverage
checks, minimum-score gates, and report provenance. Workflow-specific graders,
gold validation, and metric aggregation stay in their respective modules.

The former mode-switching `EvaluationFramework` has been removed. There is no
console command or executable package module; call the Python APIs directly.
Input loading, report persistence, credentials, and application exit handling
belong to your application.

See the [separate Python samples](samples/document-evaluation/README.md).

> [!IMPORTANT]
> Semantic evaluation and Foundry publishing send document data to the configured
> Azure services. Obtain data-owner approval first. Foundry's own telemetry and
> service behavior remain governed by the SDK and your Azure configuration.

## Comparison policy

Configuration uses JSON Pointer patterns such as
`/segments/*/fields/items/*/quantity`. An asterisk addresses existing array
occurrences; it cannot invent omitted rows.

* `fields` configures exact, decimal, date, or whitespace/Unicode-normalized text
  comparison. Text normalization is opt-in and case-sensitive.
* Decimal tolerances and unit conversions use explicit decimal strings.
  Numeric strings are rejected unless `coerce_numeric_strings` is enabled.
* `unit_field` references a sibling or absolute pointer. `unit_scales` maps every
  accepted unit to a positive conversion factor. No missing unit is guessed.
* `tables` aligns rows or document instances by independent identity keys,
  preserving duplicate occurrences. Empty `keys` selects positional comparison.
* `extraction_schema` is JSON Schema over canonical values before comparison
  normalization. Network schema references are not accepted.
* `OnlineEvaluationConfig` owns `model`, `groundedness`, `field_labels`,
  `completeness`, grader instructions, and the field-count budget.
* `OfflineEvaluationConfig` owns `text_f1`, optional text-overlap diagnostics,
  not extraction F1. It rejects online-only settings rather than ignoring them.
* `thresholds` maps metric names to minimum scores. Missing or undefined metrics
  fail. Pooled `extraction.micro_f1` is computed from occurrence counts, not the
  mean of per-document F1 values.
* Context and field-count budgets reject oversize assessments rather than
  truncating evidence and pretending coverage is complete.
* `project` optionally enables Foundry publishing.

Gold annotations use the adapter's canonical structure:

```json
{
  "segments": [
    {
      "category": "Events",
      "pages": [1, 1],
      "fields": {"items": [{"eventId": "A", "quantity": 12}]}
    }
  ]
}
```

Use `CUAdapter(config).prepare(raw, document_id)["response"]` to inspect the
format, but annotate gold independently from original evidence. Wrong nonempty
values count as both a false positive and a false negative. Category and page
metadata affect exact match but do not inflate field F1. Tolerant scores remain
separate from strict normalized exact match.

## Reports and failure behavior

Reports include `metrics`, `fields`, `gates`, `gate_status`, configuration hash,
package/SDK versions, and native job results under `foundry`. The latter retain
input evidence and evaluator output/reasons.

Gate status is `pass`, `fail`, or `not_configured`. An unconfigured gate is not a
release pass. Empty batches, duplicate IDs, missing offline gold, provider
errors, partial row coverage, and unusable evaluator outputs abort the run.
Library provider exceptions propagate unchanged. Methods return report
dictionaries; the package does not write report files or convert failures into
process exit codes.

## Extend without changing the library

Supply additional Foundry-compatible evaluator callables:

```python
def check_application_rule(*, response: str, **kwargs: object) -> dict[str, object]:
    """Assess an application-owned rule."""
    return {"within_size_limit": int(len(response) <= 10000)}

evaluator = OfflineEvaluator(
    config, evaluators={"application_rule": check_application_rule}
)
```

The SDK inspects signatures, so name variadic keyword arguments `kwargs`.
Custom evaluators cannot replace reserved built-ins. An optional `runner`
constructor argument on either evaluator accepts a caller-configured function compatible with
Foundry's `evaluate` for controlled execution/testing. All such dependencies
remain caller-owned.

## Scope and limits

Extractly evaluates extraction; it does not run CU, ingest original documents,
provision Azure, own storage, implement business arithmetic, or manage review
workflows. Callers retain original bytes, source identities, and raw responses.

The five field labels are `supported`, `unsupported`, `conflicting`, `missing`,
and `ambiguous`. Completeness grading independently searches for omissions,
including missing rows and instances; it is not proven source-level recall.
Grader results do not provide independently verified source citations for
every judgment. Shared OCR errors can fool both extraction and evaluation.

`p_error` is always null. Failing deterministic or semantic checks recommend
review; other results remain `uncalibrated`. Neither a high score nor a passing
gate automatically approves a business document. Golden-data curation,
calibration, statistical certification, human review, and deployment decisions
remain application responsibilities.

## Development

```powershell
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy src\extractly
uv build
```

Tests use synthetic fixtures and mock semantic service boundaries. SDK-marked
tests exercise real local Foundry execution without model requests. Live model
grading and publishing require authorized Azure resources and are not certified
by the local suite.

See [architecture](architecture.md), [project structure](Project-Structure.md),
and [contributing](CONTRIBUTING.md).

## SDK references

* [Azure AI Evaluation SDK](https://learn.microsoft.com/python/api/overview/azure/ai-evaluation-readme)
* [Dataset evaluation](https://learn.microsoft.com/azure/ai-foundry/how-to/develop/evaluate-sdk?view=foundry-classic)
