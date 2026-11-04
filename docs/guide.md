---
title: Extractly user guide
description: Evaluator contracts, injectable pipelines, client utilities, and metric aggregation.
---

[README](../README.md) | [Architecture](../architecture.md) |
[Project structure](../Project-Structure.md)

## Usage

Use deterministic field and structural evaluators against independent gold,
or LLM field and structural evaluators against source Markdown and original
analyzer rules. Schema validation remains an independent check. All evaluators
return a nonempty `metrics` dictionary; other keys are diagnostics.

The [README quick start](../README.md#quick-start) is a complete offline
example. Pipelines accept saved CU dictionaries; standalone evaluators accept
normalized fields rather than CU wrappers or canonical segment envelopes.

## Comparison models

`EvaluationModel` declares selected typed fields, aliases, constraints, and
registered comparisons. Nested evaluation models describe structured rows.
Defaults describe optionality without manufacturing observed values.

```python
"""Compare a quantity under a caller-defined tolerance policy."""

from decimal import Decimal

from extractly import EvaluationModel, FieldEvaluator, field_comparator


class Quantity(EvaluationModel):
    """Declare a decimal quantity and its accepted variation."""

    quantity: Decimal

    @staticmethod
    @field_comparator("quantity")
    def within_one_tenth(expected: Decimal, actual: Decimal) -> bool:
        """Accept values that differ by at most one tenth."""
        return abs(expected - actual) <= Decimal("0.1")


result = FieldEvaluator()({"quantity": "10.05"}, {"quantity": 10}, Quantity)
assert result["metrics"]["extraction.document.tp"] == 1
```

Place `@staticmethod` above `@field_comparator`. Comparison functions receive
validated expected and actual values and must return exactly `bool`. They
are comparison policies, not transforming Pydantic validators. Caller errors
propagate unchanged. Invalid gold raises; invalid present predictions count
as errors. A wrong present value contributes both FP and FN.

Missing keys, null, and empty strings are absence; zero and false are present.
Arrays compare positionally. A whole-container comparator owns its field
comparison but cannot suppress the separate structural evaluator's counts.

## Standalone evaluators

| Evaluator | Call parameters | Metrics |
| --- | --- | --- |
| `FieldEvaluator()` | `extracted_values`, `ground_truth`, `extraction_model` | `extraction.document` TP/FP/FN and ratios |
| `StructuralEvaluator()` | Same three call parameters | `extraction.table_row` and `extraction.table_count` TP/FP/FN and ratios |
| `SchemaEvaluator()` | `extracted_values`, `extraction_model` | `valid`, `violations` |
| `LLMFieldEvaluator(client, deployment)` | `markdown`, `extracted_values`, `analyzer_definition`, `extraction_model` | Field counts, TN, ambiguity, and ratios |
| `LLMStructuralEvaluator(client, deployment)` | Same four call parameters as LLM field evaluation | Source-assisted table/row counts and ratios |

Field evaluation also returns field-only `exact_match`, `field_counts`, and
concrete `comparisons`. Structural evaluation returns per-table diagnostics.
LLM evaluators return judgments, deployment, latency, and usage. Structural
LLM judgments assess missing/extra/duplicate rows, ordering, merging,
splitting, table identity, and row/table association.

Each model-selected array of structured object records is one logical table,
including an empty array. Missing, null, and non-list containers are not
tables. Scalar arrays are not tables. Discovery handles aliases, nested
objects, nullable definitions, and local references. Unsupported recursive
or ambiguous container unions raise rather than silently changing scope.

Matching counts do not establish row identity, order, or scalar correctness.
Consult structural judgments and review flags separately. Undefined ratios
remain `None`; defined ratios are rounded to three decimal places.

## Offline evaluation

`OfflineEvaluationPipeline(config, runner=..., evaluators=None)` selects:

* `field`: `FieldEvaluator()`
* `structural`: `StructuralEvaluator()`
* `schema`: `SchemaEvaluator()`

Supply `OfflineEvaluationConfig(extraction_model=YourModel)`. The default
structural evaluator needs a model containing a record-array table. For
scalar-only extraction, explicitly select field and schema checks.

`run(raw, document_id=..., ground_truth=...)` evaluates one document.
`_evaluate(documents)` is an internal batch helper for records containing `id`, `cu_result`, and
`ground_truth`. Gold may be a normalized fields dictionary for one segment,
or canonical `{"segments": [{"fields": ...}]}` annotations. Multiple
segments require canonical gold so positions remain explicit. Gold must be
independent of the predictions.

### Golden datasets in Azure Blob Storage

`extractly.store.GoldenDatasetStore` defines `load(prefix)`, returning a
dictionary of independent ground-truth objects keyed by document ID.
`AzureBlobGoldenDatasetStore` implements it with a caller-configured,
synchronous `azure.storage.blob.ContainerClient`. Both classes are also
available from `extractly`. The store never creates, authenticates, or closes
the client, and does not perform evaluation.

Keep one UTF-8 JSON object per document beneath a dataset directory prefix:

```text
gold/v1/delivery-001.json
gold/v1/archive/delivery-002.json
```

Calling `load("gold/v1")` or `load("gold/v1/")` returns annotations keyed by
`delivery-001` and `archive/delivery-002`. Only the final `.json` extension
is removed, case-insensitively; nested paths and other extensions are
preserved. An empty prefix selects the entire container. Non-JSON blobs are
ignored. UTF-8 byte-order marks are accepted. Each JSON object can contain
normalized fields or canonical segment annotations and is returned unchanged;
evaluator-specific schema validation remains separate.

The following example requires a caller-supplied `container_client`, an
extraction `model` containing a record-array table, and
`cu_results`, a dictionary of saved CU responses keyed by the same document
IDs. It makes Blob Storage list and download requests:

```python
from extractly import OfflineEvaluationConfig, OfflineEvaluationPipeline
from extractly.eval_runner import LocalRunner
from extractly.store import AzureBlobGoldenDatasetStore, GoldenDatasetStore

store: GoldenDatasetStore = AzureBlobGoldenDatasetStore(container_client)
gold = store.load("gold/v1/")
documents = [
    {"id": document_id, "cu_result": cu_results[document_id], "ground_truth": annotation}
    for document_id, annotation in gold.items()
]
pipeline = OfflineEvaluationPipeline(
    OfflineEvaluationConfig(extraction_model=model),
    runner=LocalRunner(),
)
report = pipeline.run(documents)
```

This example runs the default field, structural, and schema evaluators against
the loaded gold. Loading holds the complete annotation dataset in memory. An empty
match, invalid JSON object, empty document ID, or ID collision raises
`ValueError`; no partial dataset is returned. Azure exceptions propagate
unchanged. Authentication, retry policy, timeouts, and client cleanup remain
the caller's responsibility.

## Online evaluation

`OnlineEvaluationPipeline` defaults to LLM field, LLM structural, and schema
checks. Supply an already-configured synchronous OpenAI client and deployment.
The following example requires caller-supplied `client`, `raw`, `model`, and
`analyzer_definition`; it makes remote inference requests:

```python
from extractly import OnlineEvaluationConfig, OnlineEvaluationPipeline
from extractly.eval_runner import LocalRunner

pipeline = OnlineEvaluationPipeline(
    OnlineEvaluationConfig(
        extraction_model=model,
        thresholds={"field.extraction.document.f1": 0.95, "schema.valid": 1.0},
    ),
    runner=LocalRunner(),
    client=client,
    deployment="your-deployment",
)
report = pipeline.run(
    raw,
    document_id="document-001",
    analyzer_definition=analyzer_definition,
)
```

Authentication, retries, timeouts, and cleanup remain client-owned.
`LocalRunner` does not make LLM evaluators local. Source Markdown must cover
the judged scope; incomplete evidence can produce unknown or ambiguous scores.

Both `run(raw, document_id=..., analyzer_definition=...)` and its
`evaluate(...)` alias accept exactly one CU result dictionary. Lists,
generators, and other batch inputs are rejected before runner execution.
Neither method has a gold parameter. Incoming `ground_truth` in lower-level
records is ignored and never enters runner datasets or online evaluator inputs.

An explicit evaluator mapping replaces defaults and does not require `client`
or `deployment`. For schema-only assessment, select
`evaluators={"schema": SchemaEvaluator()}` and supply an extraction model.

## Extending Extractly

Pass nonempty evaluator names without dots mapped to synchronous callables.
The mapping replaces defaults. Each callable receives its declared subset of
normalized call parameters; `**kwargs` accepts all available parameters.

```python
"""Define an application-owned metric-returning evaluator."""

from extractly.types import JsonObject


def application_check(extracted_values: JsonObject) -> JsonObject:
    """Report an application-defined score and separate diagnostics."""
    return {"metrics": {"score": float(bool(extracted_values))}, "reason": "Application policy"}
```

Select it using `evaluators={"application": application_check}`. Metric
values must be finite numbers, booleans, or `None`. Bare scores, strings,
nested metric values, and nonfinite numbers are rejected. Diagnostics outside
`metrics` are not aggregated or retained in compact reports.

## Reports and metrics

### Offline reports

Offline run and document metrics are grouped by the configured evaluator names:

```json
{
  "metrics": {
    "field": {"extraction.pooled_field_f1": 1.0},
    "structural": {"extraction.table_row.f1": 1.0},
    "schema": {"valid": 1.0, "violations": 0.0}
  },
  "document_metrics": [
    {"id": "document-001", "metrics": {"schema": {"valid": 1}}, "review_required": false}
  ]
}
```

This abbreviated example omits additional metrics. Offline reports also include mode,
document count, gates, gate status, and a configuration hash. Raw source values,
comparison diagnostics, and native runner rows are omitted.

### Online reports

Online reports contain direct metrics for the one evaluated document:

```json
{
  "mode": "online",
  "id": "document-001",
  "metrics": {
    "field": {"extraction.document.f1": 1.0},
    "structural": {"extraction.table_row.f1": 1.0},
    "schema": {"valid": 1, "violations": 0}
  },
  "review_required": false,
  "configuration_hash": "<configuration fingerprint>"
}
```

The metric dictionaries above are abbreviated. There is no `documents` count,
`document_metrics` list, cross-document mean, pooled score, `gates`, or
`gate_status`. A dedicated online report builder reads document row metrics,
not the runner's batch aggregates or the offline report builder.

`review_required` combines evaluator-supplied review signals, failed schema
conformance, unresolved counts, directional errors, and failures of configured
minimum-score thresholds. Online threshold keys name document metrics, such as
`field.extraction.document.f1`, not `field.extraction.pooled_field_f1`.
Missing, undefined, and boolean scores fail numeric thresholds. Thresholds use
unrounded scores before the report is rounded to three decimal places. With
no configured thresholds, document-review signals alone determine the flag.
The flag is a model-assisted review signal, not independently verified accuracy.

### Offline aggregation

Document scores are averaged over defined numeric values. All-undefined
scores remain `None`. Field occurrence counts additionally produce
`extraction.pooled_field_precision`, `extraction.pooled_field_recall`, and
`extraction.pooled_field_f1` from summed TP/FP/FN. Corresponding document means
use `extraction.mean_document_*` names. Other complete TP/FP/FN groups expose
`pooled_precision`, `pooled_recall`, and `pooled_f1` under their metric prefix.
Unresolved occurrence groups retain unknown pooled ratios.

Canonical segments are evaluated at their original positions. Within one
document, segment occurrence counts are summed and ratios recomputed. Schema
validity requires every evaluated segment to conform. Document review flags
remain separate from batch thresholds and incorporate evaluator-supplied
review signals, failed conformance, unresolved counts, and directional errors.

### Offline minimum-score gates

Threshold keys combine evaluator and metric names, such as
`field.extraction.pooled_field_f1` or `schema.valid`. Missing, undefined, and
boolean scores fail numeric gates. Gates use unrounded values, then report
floats are rounded to three decimal places. No thresholds gives
`gate_status="not_configured"`. Applications own business approval and report
persistence. Model-assisted scores are not independently verified accuracy.

## CU parsing

`parse_cu_result` returns a `ParsedCUResult` containing an ordered
`list[ParsedCUSegment]` and `string_encoding`. Each frozen segment dataclass
has `content_path`, `category`, `pages`, `markdown`, and `fields` attributes.
There is no evidence collection or separate response/Markdown map.

```python
"""Access typed parsed segments and serialize them with standard dataclass tools."""

from dataclasses import asdict

from extractly.cu import ParsedCUSegment, parse_cu_result

raw = {"contents": [{
  "path": "input1",
  "markdown": "Quantity: 10",
  "fields": {"quantity": {"valueInteger": 10}},
}]}
parsed = parse_cu_result(raw)
segment = parsed.segments[0]

assert isinstance(segment, ParsedCUSegment)
assert segment.markdown == "Quantity: 10"
assert segment.fields == {"quantity": 10}
assert asdict(parsed)["segments"][0]["fields"] == {"quantity": 10}
```

Dataclass attribute references are frozen; nested dictionaries and page lists
remain mutable. The parser does not collect CU confidence, geometry, or spans.
Classifier-declared segments without child extraction content retain empty
Markdown and fields, rather than borrowing the parent's full text.

Both pipelines call this parser and serialize its segments inside a
`{"segments": [...]}` envelope. Each evaluator receives only its owning
segment's Markdown. Combined document context is
retained for budgeting, not substituted for segment-local text. Existing
`/segments/{index}/fields/...` pointers stay unchanged. Segment-local text
does not establish complete-document coverage or cross-segment table identity.

The CU package contains only the parser module and its exports. The adapter,
analyzer/page helpers, and ground-truth draft builder have been removed.
Supply independently reviewed normalized fields or canonical segments as
offline ground truth. Structural scope comes from the extraction model.

`EvaluationModel`, `field_comparator`, `ModelContract`, and comparison-version
metadata remain supported. Static comparison methods are essential to
deterministic field evaluation; LLM judgments do not execute those methods.
Ground truth must be checked against the original source before use as gold.

## Execution runners

Both pipelines require explicit `runner=` injection. `LocalRunner` invokes
callables in process. `FoundryEvaluationRunner` retains SDK execution and
optional publishing through a configured destination or caller-owned client.
Both consume `RunnerRequest` and return `RunnerResult` instances.

Temporary JSONL inputs are cleaned up on success and failure. This is cleanup,
not secure erasure. SDK telemetry and remote handling remain subject to the
supplied providers' policies. Provider exceptions propagate unchanged.

## Migration

Replace the old `OfflineEvaluator` and `OnlineEvaluator` names with the
corresponding pipeline classes. The combined extraction evaluator, Decision
evaluator family, and text-F1 path were removed. Custom evaluators must return
explicit metrics. Report consumers use
`report["metrics"][evaluator_name][metric_name]`. Replace CU helper imports
with `parse_cu_result`; custom comparison utilities remain supported.

Parsed-result consumers now use `parsed.segments` and dataclass attributes,
not `parsed.response`, `parsed.markdown`, or `parsed.evidence`. Adapter records
likewise no longer contain top-level evidence or Markdown maps; local Markdown
is stored on each segment inside the serialized response.

## Local development setup

Use the root `.venv` and uv-managed dependencies:

```powershell
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy src
uv build
```

Tests use synthetic documents and mocked model boundaries. Live inference and
publishing need separate authorization. Existing exploratory notebooks and
generated results are not migrated automatically and may use removed APIs.