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
* Measure field precision/recall/F1, exact match, table-row detection, and table counts
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
from datetime import date
from decimal import Decimal
from pathlib import Path

from extractly import EvaluationModel, OfflineEvaluationConfig, OfflineEvaluator
from extractly.eval_runner import LocalRunner


class LineItem(EvaluationModel):
  """Select quantity and optional ISO delivery dates."""

  quantity: Decimal
  deliveryDate: date | None = None


class DeliveryEvaluation(EvaluationModel):
  """Declare scored fields and their validation in one place."""

  deliveryMode: str
  unit: str
  lineItems: list[LineItem]


sample = Path("samples") / "document-evaluation"
config = OfflineEvaluationConfig(extraction_model=DeliveryEvaluation)
document = json.loads((sample / "synthetic.jsonl").read_text(encoding="utf-8"))

evaluator = OfflineEvaluator(config, runner=LocalRunner())
report = evaluator.evaluate(
    document["cu_result"],
    document_id=document["id"],
    ground_truth=document["ground_truth"],
)
assert report["metrics"]["extraction.pooled_field_f1"] == 1.0
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
from extractly import FoundryEvaluationRunner, OnlineEvaluationConfig, OnlineEvaluator

online = OnlineEvaluator(OnlineEvaluationConfig(extraction_model=DeliveryEvaluation), runner=FoundryEvaluationRunner())
online_report = online.evaluate(document["cu_result"], document_id=document["id"])
```

For semantic checks, pass your application's configured credential:

```python
online_settings = json.loads((sample / "online.json").read_text(encoding="utf-8"))
online_config = OnlineEvaluationConfig.model_validate({**online_settings, "extraction_model": DeliveryEvaluation})
semantic_evaluator = OnlineEvaluator(online_config, runner=FoundryEvaluationRunner(), credential=credential)
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

`extraction_model` accepts an `EvaluationModel` class describing each canonical
segment's `fields` object. That one model supplies field selection, validation,
and comparison. There is no separate field-rule configuration or path allowlist.
The read-only `config.extraction_schema` property exposes its compiled evaluation
contract; this is derived automatically and is not a configuration input.

* Declared fields are scored; undeclared values and metadata are not scored.
  Nested models and lists select nested occurrences without rewriting CU or gold.
* Pydantic types, aliases, requiredness, and `Field` constraints validate inputs.
  Standard Pydantic type conversion applies; use strict field types when needed.
* Validated values use equality unless a registered `@field_comparator` supplies
  domain-specific normalization, tolerances, or parsing. Methods have arbitrary
  names and return exactly one `bool`: `True` for a match or `False` for a
  mismatch. Allowed variation counts as a normal match in every extraction
  score, including `exact_match`. Non-boolean results raise `TypeError`.
* Missing values are handled by the engine, not comparator callbacks. Defaults
  never manufacture observed values. Invalid gold raises; invalid predictions
  produce schema violations and remain visible in directional error counts.
* Arrays are always compared in positional order; reordered values are errors.
  `tables` selects structural metrics for configured array paths, preserving
  duplicate occurrences. `TableRule(level="table_row")` is the default and emits
  `table_row_*` metrics. `TableRule(level="table_count")` emits `table_count_*`
  metrics for arrays representing tables, such as `/segments`. Identity keys
  and the previous `row` and `instance` level names are not accepted.
* Without an extraction model, the default is equality over all emitted values,
  with no additional model-validation constraints.
* `OnlineEvaluationConfig` owns `model`, `groundedness`, `field_labels`,
  `completeness`, grader instructions, and the field-count budget.
* `OfflineEvaluationConfig` owns `text_f1`, optional text-overlap diagnostics,
  not extraction F1. It rejects online-only settings rather than ignoring them.
* `thresholds` maps metric names to minimum scores. Missing or undefined metrics
  fail. Run-level `extraction.mean_document_precision`,
  `extraction.mean_document_recall`, and `extraction.mean_document_f1` average
  defined per-document scores. `extraction.pooled_field_precision`,
  `extraction.pooled_field_recall`, and `extraction.pooled_field_f1` are computed
  from summed field occurrence counts, not document averages. Individual
  `document_metrics` entries keep `extraction.precision`, `extraction.recall`,
  and `extraction.f1`.
* Run-level `extraction.exact_match` is `1` only when every document matches,
  otherwise `0`. `extraction.document_exact_match_rate` reports the fraction
  of matching documents, rounded to three decimal places.
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

### Select fields to evaluate

Keep complete CU results and gold annotations, then declare the fields that matter.
Extend the line-item model above to add a caller-owned tolerance:

```python
from extractly import field_comparator


class QuantityLineItem(LineItem):
    """Add quantity tolerance without a second configuration structure."""

    @staticmethod
    @field_comparator("quantity")
    def close_enough(expected: Decimal, actual: Decimal) -> bool:
      """Accept a quantity difference of at most 0.1 as a match."""
      return abs(expected - actual) <= Decimal("0.1")


  class QuantityDelivery(DeliveryEvaluation):
    """Use the line-item comparison policy for every source row."""

    lineItems: list[QuantityLineItem]

config = OfflineEvaluationConfig(
  extraction_model=QuantityDelivery,
  thresholds={"extraction.pooled_field_f1": 0.95},
)
report = OfflineEvaluator(config, runner=LocalRunner()).run([document])
```

Missing, extra, or incorrect selected occurrences still count, including duplicate
rows. Undeclared identity fields remain available to `tables` without being scored.
Table paths use JSON Pointer escapes (`~1` for `/`, `~0` for `~`); `*` matches one
array occurrence. Domain rules that need several values can register a comparator
on a nested model field to compare the whole object as one scored value.

For a non-ISO date format, declare a constrained string and parse it inside a
registered comparator. The engine does not impose a date-string convention.
Comparison callbacks are synchronous static methods accepting expected and actual
values. Custom Pydantic transformation validators, recursive models, and ambiguous
unions of nested containers are unsupported.

Configuration JSON describes the model schema and comparator identities, but cannot
reload executable Python classes. Python-mode configuration copies retain the model.
Bump the model's `comparison_version` when its comparison policy changes so report
configuration hashes distinguish policy revisions.

Custom evaluators, optional text-F1 diagnostics, and document-wide online
groundedness and completeness checks still receive the full extraction and remain
independent. Selection never changes annotations, predictions, or evidence budgets.

### Draft ground truth from CU

Use `build_ground_truth` to create an editable dictionary in this format from
either raw CU content or a successful operation envelope:

```python
from extractly import build_ground_truth

draft = build_ground_truth(document["cu_result"])
```

The helper requires no evaluation configuration, credentials, or document ID.
It shares CU parsing with `CUAdapter`, preserves nested values and duplicate
occurrences, and excludes confidence, spans, source metadata, and Markdown.
Editing the draft does not mutate the raw CU result. File persistence remains
the caller's responsibility.

> [!WARNING]
> A CU-derived draft is not independent gold. Verify and correct it against the
> original document, including missing fields, rows, and document instances,
> before passing it as `ground_truth`. Comparing unchanged predictions with
> their own draft cannot establish extraction accuracy.

CU conversion lives in `extractly.cu`, with separate adapter and ground-truth
modules backed by a shared parser. Import the adapter with
`from extractly import CUAdapter` or `from extractly.cu import CUAdapter`.

Wrong nonempty values count as both a false positive and a false negative.
Category and page metadata affect exact match but do not inflate field F1.
Every field comparison has one boolean outcome under its configured policy;
there is no separate tolerant score set.

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

Supply additional evaluator callables or caller-configured `AzureOpenAIGrader`
objects:

```python
def check_application_rule(*, response: str, **kwargs: object) -> dict[str, object]:
    """Assess an application-owned rule."""
    return {"within_size_limit": int(len(response) <= 10000)}

evaluator = OfflineEvaluator(
  config, runner=FoundryEvaluationRunner(), evaluators={"application_rule": check_application_rule}
)
```

The SDK inspects signatures, so name variadic keyword arguments `kwargs`.
Custom evaluators cannot replace reserved built-ins. All such dependencies
remain caller-owned.

The shared `Evaluator` type is `Callable[..., JsonObject] | AzureOpenAIGrader`.
Custom callables return named-result dictionaries; grader objects are passed
unchanged to the runner. Custom runners that directly invoke evaluators must
handle or explicitly reject grader objects rather than assume every evaluator
is callable.

### Typed execution runners

Both workflow constructors require an explicit `runner` instance. Use
`FoundryEvaluationRunner()` for standard execution with
`azure.ai.evaluation.evaluate`; neither workflow silently selects a backend.
Passing `None` or a plain function is not supported.

The `extractly.eval_runner` subpackage exports `EvaluationRunner`,
`RunnerRequest`, `RunnerResult`, and `FoundryEvaluationRunner`.
Custom runners subclass `EvaluationRunner` and implement
`run(request: RunnerRequest) -> RunnerResult`.

* `RunnerRequest` contains a `Path` to the temporary JSONL dataset, a mapping
  of named evaluators, `evaluation_name`, optional `azure_ai_project`, and
  `fail_on_evaluator_errors`.
* `RunnerResult` contains `rows`, `metrics`, and optional provider `metadata`.
  Rows use `inputs.id` and `outputs.<evaluator>.<metric>` keys; metric names use
  `<evaluator>.<metric>`. Metadata cannot override the typed rows or metrics.

Runners must consume the dataset before returning and propagate provider
failures. Shared execution still validates exact ID coverage, evaluator outputs,
and schema flags; each workflow validates its additional result requirements.
The caller retains ownership of runners, evaluators, and credentials.

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
