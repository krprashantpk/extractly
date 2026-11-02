---
title: Extractly guide
description: Installation, evaluation workflows, component contracts, metrics, and local development for Extractly.
---

[Back to the README](../README.md)

## What is Extractly

Extractly evaluates saved Azure Content Understanding (CU) extraction results.
Compare predictions against independent gold annotations, or assess schema and
source support without gold. Adaptation preserves source evidence for scoring;
reports return field-occurrence metrics, score gates, and a configuration hash.
Decision evaluation also returns detailed page judgments and coverage.

```text
Saved CU result --> CUAdapter --> Canonical extraction + evidence
                                       |
                  +--------------------+--------------------+
                  |                                         |
          OfflineEvaluator                           OnlineEvaluator
          Independent gold                            No gold labels
          Typed comparisons                     Schema + optional judging
                  |                                         |
                  +---------- Explicit runner --------------+
                                       |
                          Metrics + gates + report
```

The workflows are independent. Neither invokes the other, and omitting gold
does not switch an offline evaluation into an online one. Your application owns
input loading, resource configuration, credentials, clients, retries, report
persistence, and any business approval.

### Getting started

* [Install Extractly](#installation) in an existing Python application
* [Run a local comparison](#evaluate-extraction-without-azure) without credentials
* [Select fields and comparison policy](#comparison-models)
* [Assess extraction without gold](#online-evaluation)
* [Choose an execution runner](#execution-runners)
* [Read metrics and gates](#reports-and-metrics)
* [Extend Extractly](#extending-extractly) with application-owned checks
* [Set up local development](#local-development-setup)

## Requirements

* Python 3.12 or newer
* [uv](https://docs.astral.sh/uv/getting-started/installation/) for the commands below
* Git when installing from a Git URL

Deterministic comparison and schema-only assessment with `LocalRunner` do not
require Azure credentials. Semantic judging and cloud publishing require your
own authorized services. Installing Extractly does not provision them.

## Installation

From an existing uv-managed project, add a local checkout:

```powershell
uv add C:\path\to\Extractly
```

Or install from Git:

```powershell
uv add "extractly @ git+https://github.com/krprashantpk/extractly.git"
```

Append `@<tag-or-commit>` to the URL to pin a source revision.

For library development, use [local development setup](#local-development-setup)
instead. The distribution and import names are both `extractly`. The package
has not been published to PyPI; do not assume `uv add extractly` installs this
repository. Declared dependencies include Azure AI Evaluation, OpenAI,
Pydantic, JSON Schema, and the TypeSafe SDK; there are no provider extras.

Verify the installation:

```powershell
uv run python -c "import extractly; print(extractly.__version__)"
```

## Usage

### Evaluate extraction without Azure

Run the following from an application with Extractly installed. The CU response
and gold are synthetic, so no files, credentials, or service calls are needed:

```python
"""Compare a synthetic extraction with independently written gold."""

from decimal import Decimal

from extractly import EvaluationModel, OfflineEvaluationConfig, OfflineEvaluator
from extractly.eval_runner import LocalRunner


class Invoice(EvaluationModel):
    """Select the invoice number and total for scoring."""

    invoiceNumber: str
    total: Decimal


raw = {
    "stringEncoding": "codePoint",
    "contents": [{
        "category": "Invoice",
        "startPageNumber": 1,
        "endPageNumber": 1,
        "markdown": "Invoice INV-001\nTotal: 125.50",
        "fields": {
            "invoiceNumber": {"type": "string", "valueString": "INV-001"},
            "total": {"type": "number", "valueNumber": 125.5},
        },
    }],
}
gold = {
    "segments": [{
        "category": "Invoice",
        "pages": [1, 1],
        "fields": {"invoiceNumber": "INV-001", "total": 125.5},
    }],
}
config = OfflineEvaluationConfig(
    extraction_model=Invoice,
    thresholds={"extraction.pooled_field_f1": 0.95},
)
evaluator = OfflineEvaluator(config, runner=LocalRunner())
report = evaluator.evaluate(raw, document_id="invoice-001", ground_truth=gold)

assert report["metrics"]["extraction.pooled_field_f1"] == 1.0
assert report["gate_status"] == "pass"
print(report["metrics"]["extraction.pooled_field_f1"])
print(report["gate_status"])
```

Expected output:

```text
1.0
pass
```

`extraction_model` describes each canonical segment's `fields` object, not the
whole CU response. Declared fields are selected for scoring; undeclared fields
and source evidence remain available without being scored. There is no implicit
execution backend: both workflows require `runner=`.

In real evaluations, gold must be independently verified against the original
document. A matching synthetic example demonstrates API usage, not accuracy on
your documents.

### Evaluate a batch

Reuse `raw`, `gold`, and `evaluator` from the preceding example:

```python
documents = [{"id": "invoice-001", "cu_result": raw, "ground_truth": gold}]
report = evaluator.run(documents)
assert len(report["document_metrics"]) == 1
```

Each offline record requires a unique, nonempty `id`, a `cu_result` dictionary,
and a canonical `ground_truth` dictionary. Empty batches and missing gold raise
instead of producing a passing report.

### Load saved results

Input loading and report storage belong to your application. For an existing
record containing `id`, `cu_result`, and `ground_truth`:

```python
import json
from pathlib import Path

document = json.loads(Path("document.json").read_bytes())
report = evaluator.run([document])
Path("report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
```

The base workflow report excludes raw job inputs and detailed evaluator rows.
Decision reports additionally contain page judgments and candidate values;
standalone runner results can contain full inputs. Apply your application's
access controls and retention policy to each artifact.

## Comparison models

`EvaluationModel` is the single declaration for field selection, Pydantic
validation, and comparison. Use nested models and lists to select nested
occurrences without rewriting CU or gold. Pydantic aliases, requiredness, types,
and `Field` constraints apply. Standard type conversion is allowed; choose
strict field types when conversion would hide a problem.

Validated values compare with equality unless a registered comparator supplies
a domain-specific rule. Extend `Invoice` from the quick start to accept a total
difference of at most `0.01`:

```python
from extractly import field_comparator


class RoundedInvoice(Invoice):
    """Apply an explicit tolerance to the selected invoice total."""

    @staticmethod
    @field_comparator("total")
    def totals_match(expected: Decimal, actual: Decimal) -> bool:
        """Accept a difference of at most one cent."""
        return abs(expected - actual) <= Decimal("0.01")


tolerant_config = OfflineEvaluationConfig(extraction_model=RoundedInvoice)
tolerant_report = OfflineEvaluator(tolerant_config, runner=LocalRunner()).evaluate(raw, document_id="invoice-001", ground_truth=gold)
assert tolerant_report["metrics"]["extraction.exact_match"] == 1
```

Comparators are synchronous static methods accepting expected and actual
validated values. Their names are arbitrary, but their results must be exactly
one `bool`; other results raise `TypeError`. Allowed variation counts as an
ordinary match, including in `exact_match`, not as a separate tolerant score.

* Missing values are handled by the comparison engine, not comparator callbacks.
* Defaults never manufacture an observed value.
* Invalid gold raises; invalid predictions remain errors and schema violations.
* Undeclared fields are not scored when an extraction model is supplied.
* Without a model, all emitted values are compared with default equality.
* Arrays compare at the same positional indices; reordered rows are errors.

A comparator on a nested model field can own the whole object's comparison as
one scored value. For non-ISO dates, declare a constrained string and parse it
inside a comparator. Custom Pydantic transformation validators, recursive
models, and ambiguous unions of nested containers are unsupported.

`config.extraction_schema` is a read-only compiled contract, not another
configuration input. JSON serialization describes the model and comparator
identities; it cannot reload executable classes. Python-mode configuration
copies retain the model. Bump a model's `comparison_version` when its policy
changes so configuration hashes distinguish revisions.

### Tables and row occurrences

Use `tables` to select structural metrics independently of selected field
values. This configuration selects quantities and treats each `lineItems` list
as one extracted table:

```python
from extractly import TableRule


class LineItem(EvaluationModel):
    """Select quantity within each row."""

    quantity: Decimal


class Delivery(EvaluationModel):
    """Select quantity occurrences within the line-item array."""

    lineItems: list[LineItem]


delivery_config = OfflineEvaluationConfig(
    extraction_model=Delivery,
    tables={"/segments/*/fields/lineItems": TableRule()},
)
```

`TableRule()` defaults to `level="table_row"`. It emits `table_row_*` metrics
for elements and infers `table_count_*` metrics for the selected list container.
An empty list is still a table; a missing list is not. Selecting two lists
retains two tables, rather than flattening their rows into one group.

An explicit `TableRule(level="table_count")` overrides inferred container
counts and uses array-element semantics for a separate table hierarchy, such
as `/segments`. Table paths use JSON Pointer escaping: `~1` for `/`, `~0` for
`~`, and `*` for one array occurrence. Identity-key alignment and the former
`row` and `instance` level names are not supported. Duplicates retain their
multiplicity; equal row counts do not prove row identity or association.

### Draft ground truth

`build_ground_truth` accepts raw CU content or a successful operation envelope
and returns a new editable canonical dictionary:

```python
from extractly import build_ground_truth

draft = build_ground_truth(raw)
assert draft == gold
```

No configuration, credential, or document ID is required. The helper shares
parsing with `CUAdapter`, preserves nested values and duplicate occurrences,
and removes confidence, spans, source metadata, and Markdown. Editing the
result does not mutate the input. File persistence belongs to the caller.

> [!WARNING]
> A CU-derived draft is not independent gold. Verify and correct it against the
> original document, including missing fields, rows, and instances. Comparing
> unchanged predictions with their own draft does not measure accuracy.

## Online evaluation

Online evaluation does not require gold and never runs offline comparison or
text-overlap diagnostics. Incoming batch records may contain application data;
any `ground_truth` is ignored and excluded from evaluator datasets.

### Schema-only assessment

Reuse the quick start's `Invoice` model and `raw` response:

```python
from extractly import OnlineEvaluationConfig, OnlineEvaluator

online_config = OnlineEvaluationConfig(
    extraction_model=Invoice,
    thresholds={"schema.valid": 1.0},
)
online = OnlineEvaluator(online_config, runner=LocalRunner())
online_report = online.evaluate(raw, document_id="invoice-001")
assert online_report["metrics"]["schema.valid"] == 1
assert online_report["gate_status"] == "pass"
assert not online_report["document_metrics"][0]["review_required"]
```

Schema-only evaluation is deterministic and needs no model or credentials.
A valid schema proves neither source support nor completeness. Batch records
require `id` and `cu_result`:

```python
online_report = online.run([{"id": "invoice-001", "cu_result": raw}])
```

### Decision field and structural assessment

Inject `OnlineFieldEvaluator`, `OnlineStructuralEvaluator`, or both. Supply
the original CU analyzer definition or its `fieldSchema`; it provides types,
extraction methods, descriptions, enum descriptions, and ancestor instructions.
An `EvaluationModel` selects scored fields but does not replace that analyzer.

The example below expects a saved delivery result with a `lineItems` array and
the analyzer that produced it. `client` must be your configured synchronous
`typesafe_sdk.TypeSafeClient`, with model, authentication, retries, and timeouts
already set. Calling this function makes remote Decision requests:

```python
from typesafe_sdk import TypeSafeClient

from extractly import OnlineFieldEvaluator, OnlineStructuralEvaluator
from extractly.types import JsonObject


def assess_delivery(raw_result: JsonObject, analyzer_definition: JsonObject, client: TypeSafeClient, *, complete_document: bool = False) -> JsonObject:
    """Assess saved delivery evidence through a caller-owned Decision client."""
    config = OnlineEvaluationConfig(
        extraction_model=Delivery,
        tables={"/segments/*/fields/lineItems": TableRule()},
        thresholds={"extraction.pooled_field_f1": 0.95},
    )
    evaluator = OnlineEvaluator(
        config,
        runner=LocalRunner(),
        field_evaluator=OnlineFieldEvaluator(client, min_probability=0.8),
        structural_evaluator=OnlineStructuralEvaluator(client),
    )
    previews = evaluator.prepare(raw_result, analyzer_definition=analyzer_definition, complete_document=complete_document)
    print("Prepared page requests:", len(previews))
    return evaluator.evaluate(raw_result, document_id="delivery-001", analyzer_definition=analyzer_definition, complete_document=complete_document)
```

Both evaluators must share the same client. Their questions are combined into
one sequential call per eligible physical page, in page order. Pages without
remote questions remain reportable without an empty service call. Choosing
`LocalRunner` does not make these supplied semantic evaluators local.

`prepare()` previews exact states, questions, and local prechecks without
inference. For batches, preparation of all inputs and page requests finishes
before the first Decision call. Decision records additionally require
`analyzer_definition` and may provide `complete_document` and `table_groups`.

#### Evidence and page boundaries

Set `complete_document=True` only when your application knows the CU result
covers the complete original. A page inside a multi-page result never becomes
the whole document through this flag. Multi-page Markdown requires validated,
non-overlapping `codePoint` page spans; unsupported encodings, missing spans,
and overlapping ranges raise. Oversize evidence is rejected, never truncated.
Missing, split, or out-of-range provenance defers fields and records rather
than guessing their page.

`selected_field_paths` optionally selects original paths such as
`lineItems[3].quantity`. Partial selection retains judgments without claiming
full field scores. Structural checks still examine every page-local record.
`document_level_fields` defaults to `TotalNetTonnage` and `deliveryMode`;
extend it for other global fields. Those fields are eligible only on a sole
page explicitly asserted to be the complete document.

#### Field judgments

`OnlineFieldEvaluator` owns `tp`, `fp`, `fn`, `tn`, `ambiguous`, and
`needs_document_context` judgments. A separate expected-presence question
distinguishes incorrect values (FP and FN) from extra values (FP only). True
negatives add no positive counts. Analyzer-defined absent leaves are assessed
without inventing requiredness. Closed-schema checks flag unexpected
properties and wrapper-type mismatches. Classification overlaps, priorities,
latest-date rules, and fallbacks follow the original analyzer instructions;
insufficient evidence stays uncertain.

#### Structural judgments

`OnlineStructuralEvaluator` owns eight independent checks: row counts, missing
rows, extra rows, duplicates, row association, table counts, table identity,
and row-to-table association. The output key is
`PAGE_STRUCUTURAL_INTEGRATY` (the spelling is part of the current contract).

Configured row lists preserve one table group per selected list, including
empty lists. Multiple selected top-level analyzer arrays retain distinct
request indices and original row paths. For explicit grouping, `table_groups`
maps keys such as `content_0_page_1` to lists of tables, where each table lists
original record indices. Manual groups must agree with configured boundaries;
missing and duplicate assignments are flagged locally. Without configured
lists or explicit groups, table checks remain unassessable. Grouping is never
invented from Markdown.

`record_field` can select an analyzer object-record array when row-list paths
are not configured; the default is `lineItems` or the sole object array.
Source-count choices default to zero through 100 rows and zero through 20
tables per page, plus overflow and uncertainty. Set `max_source_rows` and
`max_source_tables` on `OnlineStructuralEvaluator` to change those bounds.
Overflow is not clipped into the maximum count.

#### Coverage and failures

Decision reports set `metric_basis="model_assisted_page_evaluation"` and add
`judgments` and `coverage`. Detailed judgments retain values, verdicts,
probabilities, schema errors, deferrals, structure, and source counts. Pages
are not counted as independent documents.

Completed field evaluation computes counts from resolved judgments; uncertain
judgments remain excluded and require review. If every judgment is uncertain,
counts are zero and positive ratios are undefined. Deferred, unselected, or
unrun scope leaves aggregate field metrics null. Structural uncertainty or
failure keeps field coverage incomplete without suppressing otherwise resolved
counts. Unknown pages are never silently dropped.

Multi-page results do not assert document-wide field scores or unique-document
table counts. Those metrics remain null. Row-presence counts can pool across
pages only when all page counts are resolved. Undefined Decision gates are
null with `gate_status="incomplete"`; a failing gate takes precedence. An
unknown gate metric name raises. A gate result is never business approval.

Provider and malformed-response errors propagate. After a later page fails,
`evaluator.page_results[document_id]` retains validated earlier pages, the
failed page, and unattempted pages. `request_previews` retains prepared batch
requests by document ID. No successful final report is fabricated after a
failure. Clients, retries, and cleanup remain caller-owned.

## Execution runners

Both workflow constructors require an explicit `EvaluationRunner` instance.
`None`, plain functions, and implicit backend selection are not supported.

| Runner | Execution | Publishing |
| --- | --- | --- |
| `LocalRunner()` | Synchronous evaluator callables in this process | Not supported |
| `FoundryEvaluationRunner()` | `azure.ai.evaluation.evaluate` | Optional classic publishing through `config.project` |
| `FoundryEvaluationRunner(client=client)` | The same SDK execution, plus submission of precomputed scores | New Foundry evaluation and run through the supplied OpenAI client |

### Local execution

Import `LocalRunner` from `extractly.eval_runner`. It invokes compatible
synchronous callables sequentially and averages numeric outputs. Boolean,
NaN, and nonnumeric outputs do not become aggregate metrics. It rejects
`AzureOpenAIGrader` objects and project publishing requests; use the Foundry
runner for those. Evaluator callables can still make external calls, so inspect
your supplied evaluators before treating a run as local-only.

### Foundry execution and publishing

Use the standard SDK runner with the quick start's local evaluators:

```python
from extractly import FoundryEvaluationRunner

foundry_evaluator = OfflineEvaluator(config, runner=FoundryEvaluationRunner())
```

This selects SDK execution without enabling model requests or new Foundry
publishing. The SDK may create local artifacts or telemetry under its own
configuration. `config.project` optionally selects classic Foundry publishing;
`LocalRunner` rejects that setting.

For new Foundry publishing, pass an application-configured synchronous
`openai.OpenAI` client targeting your project's `/openai/v1` endpoint. This
function expects that client and the quick start's synthetic input:

```python
from openai import OpenAI


def publish_invoice(client: OpenAI) -> JsonObject:
    """Publish precomputed invoice scores using a caller-owned Foundry client."""
    evaluator = OfflineEvaluator(config, runner=FoundryEvaluationRunner(client=client))
    return evaluator.evaluate(raw, document_id="invoice-001", ground_truth=gold)
```

The runner submits precomputed per-document scores, not cloud rescoring.
Only metrics with finite numeric values in `[0, 1]` for every row become
grading criteria. TP, FP, FN, and violation counts remain diagnostics, not
grading criteria. Submitted records contain document IDs and scalar numeric
or boolean outputs, not source inputs or detailed evaluator diagnostics.
Workflow review flags are added after runner execution and are not published.
Portal averages are document-score means, not pooled extraction metrics.

Standalone `RunnerResult.metadata` includes `eval_id`, `run_id`, and
`report_url` after new Foundry submission. A URL may not be available until the
cloud run finishes. Workflow reports intentionally omit native runner metadata;
do not expect a `foundry` or `report_url` property in the compact report.

> [!IMPORTANT]
> Semantic evaluation sends document evidence to the configured Decision
> service. Publishing sends data to the selected Foundry service; classic SDK
> publishing may include source inputs, while new Foundry publishing submits
> IDs and scalar outputs. Obtain data-owner approval for the chosen path.
> SDK telemetry and service behavior are governed by your configuration.

## Reports and metrics

Base reports return `mode`, `documents`, `metrics`, `document_metrics`, `fields`,
`gates`, `gate_status`, and `configuration_hash`. They do not include raw
datasets, native SDK rows, or SDK version metadata. Each `document_metrics`
entry includes its `id`, evaluator-qualified scores, and `review_required`.
Review flags are document-specific and independent of run-level gates.

| Metric | Meaning |
| --- | --- |
| `schema.valid` | Mean schema conformance, with 1 for valid documents |
| `extraction.pooled_field_precision` | TP / (TP + FP), using summed field occurrences |
| `extraction.pooled_field_recall` | TP / (TP + FN), using summed field occurrences |
| `extraction.pooled_field_f1` | 2TP / (2TP + FP + FN), using summed field occurrences |
| `extraction.mean_document_precision`, `mean_document_recall`, `mean_document_f1` | Means of defined per-document extraction scores, each prefixed with `extraction.` |
| `extraction.exact_match` | 1 only when every document matches; otherwise 0 for resolved offline comparison |
| `extraction.document_exact_match_rate` | Fraction of matching documents |
| `extraction.table_row_*`, `extraction.table_count_*` | Structural occurrence counts and precision, recall, and F1 |

Per-document extraction scores use `extraction.precision`, `extraction.recall`,
and `extraction.f1`. Offline `fields` contains per-path counts and ratios;
online `fields` is empty, with Decision detail in `judgments` when enabled.
Wrong nonempty selected values contribute both FP and FN. Missing values
contribute FN; extra values contribute FP. Category and page metadata do not
inflate field F1. With an extraction model, selected fields and their containing
structure determine exact match; without one, category and page metadata can
also affect it.

Undefined positive ratios remain null, not zero or perfect scores. Empty
agreement may have exact match 1 while positive-extraction ratios are undefined.
Local aggregation omits a mean-document score if every corresponding document
score is undefined. Displayed metrics are rounded to three decimal places;
base gate checks use unrounded scores.

### Minimum-score gates

`thresholds` maps metric names to minimum acceptable values. Base workflows
return `pass`, `fail`, or `not_configured`; missing, undefined, nonnumeric,
boolean, and nonfinite scores fail configured gates. Decision evaluation has
the additional incomplete behavior described under
[coverage and failures](#coverage-and-failures).

A passing gate does not clear a document's review flag or prove accuracy
without independent gold. No configured thresholds means `not_configured` in
the base workflows, not a release pass. In the Decision path an empty gate
mapping currently yields `pass`; inspect coverage and document review flags
rather than treating that status as a configured release policy.

### Optional text overlap

`OfflineEvaluationConfig(text_f1=True)` adds the SDK's text-overlap diagnostic.
It is not the typed extraction F1 metric and does not replace field, row, or
table comparison. Online configuration rejects `text_f1`.

## Component reference

| Component | Responsibility | Main entry points |
| --- | --- | --- |
| `EvaluationModel`, `field_comparator` | Selected fields, validation, comparison policy | Model declarations and registered static methods |
| `EvaluationConfig`, `TableRule` | Shared extraction contract, table paths, budgets, gates | Validated Pydantic settings |
| `CUAdapter` | Canonical values, spans, confidence, Markdown, encoding | `prepare(raw, document_id)`, `field_rows(record)` |
| `build_ground_truth` | Editable canonical annotation draft | `build_ground_truth(raw)` |
| `SchemaEvaluator` | Validate canonical extraction | `evaluator(response=...)` |
| `ExtractionEvaluator` | Positional comparison against gold | `evaluator(response=..., ground_truth=...)` |
| `OfflineEvaluator` | Gold-bearing dataset preparation and pooled metrics | `evaluate(...)`, `run(documents)` |
| `OnlineEvaluator` | Reference-free schema and injected Decision checks | `prepare(...)`, `evaluate(...)`, `run(documents)` |
| `OnlineFieldEvaluator` | Field question construction and scoring | `prepare_page(...)`, `score(...)` |
| `OnlineStructuralEvaluator` | Independent row and table checks | `prepare_page(...)`, `score_page(...)` |
| `EvaluationRunner` | Typed execution boundary | `run(request) -> RunnerResult` |

The public root exports are listed in
[the package initializer](../src/extractly/__init__.py). Runner interfaces and
both built-in implementations are available from `extractly.eval_runner`.
Use `OfflineEvaluationConfig` or `OnlineEvaluationConfig` with its corresponding
workflow; `EvaluationConfig` is for reusable adapters and evaluators.

Common configuration includes `name`, `extraction_model`, `tables`, `project`,
`thresholds`, and `max_context_characters` (default 100,000).
Online-only configuration includes `max_fields_per_document` (default 2,000),
`selected_field_paths`, `document_level_fields`, and `record_field`.
Unknown settings are rejected. The online workflow has no `credential`,
`model`, `groundedness`, `field_labels`, or `completeness` constructor/config
options; configure semantic dependencies explicitly and inject them.

## Extending Extractly

### Application-owned evaluator callables

Supply named checks through `evaluators=` without modifying a workflow:

```python
def check_size(*, response: str, **kwargs: object) -> dict[str, object]:
    """Report whether the canonical extraction fits an application size limit."""
    return {"within_limit": int(len(response) <= 10000)}


custom = OfflineEvaluator(config, runner=LocalRunner(), evaluators={"size": check_size})
custom_report = custom.evaluate(raw, document_id="invoice-001", ground_truth=gold)
assert custom_report["metrics"]["size.within_limit"] == 1
```

Use the name `kwargs` for variadic keyword arguments because the SDK inspects
signatures. Custom checks cannot replace reserved built-in evaluator names.
Checks receive the complete canonical extraction; field selection does not
silently rewrite their inputs or reduce evidence budgets.

The shared `Evaluator` type permits `Callable[..., JsonObject]` and
caller-configured `AzureOpenAIGrader` objects. Graders require the Foundry
runner; custom runners must handle or explicitly reject them. Returned named
scores should be numeric when they need aggregation.

### Custom execution runners

Subclass `EvaluationRunner` and implement
`run(request: RunnerRequest) -> RunnerResult`. No plugin registration is needed.

* `RunnerRequest` contains the temporary JSONL `Path`, named evaluators,
  `evaluation_name`, optional `azure_ai_project`, and `fail_on_evaluator_errors`.
* `RunnerResult` contains `rows`, `metrics`, and optional provider `metadata`.
* Rows use `inputs.id` and `outputs.<evaluator>.<metric>` keys; aggregate metric
  names use `<evaluator>.<metric>`.

Consume the dataset before returning; its temporary path does not outlive the
workflow call. Propagate provider failures and preserve exact document ID
coverage. Metadata cannot override typed rows or metrics. Shared execution
validates coverage and schema outputs; workflows validate their additional
results. Keep evaluator and runner lifetimes caller-owned.

## Failure behavior and limits

Empty batches, duplicate IDs, missing offline gold, invalid gold, oversize
evidence, partial row coverage, and unusable evaluator outputs raise.
Provider exceptions propagate rather than becoming successful partial
averages. Methods return report dictionaries, not process exit codes.

Temporary datasets are removed on context exit, including on failure. This
is cleanup, not secure erasure. SDK-created artifacts and telemetry have their
own behavior and require an application retention policy.

Extractly does not run CU analysis, load original document bytes, provision
services, own storage, validate business arithmetic, calibrate model
probabilities, or implement a human review workflow. It has no console command
or executable package module and does not load environment files. The former
mode-switching `EvaluationFramework` is not available.

Schema conformance is not evidence support. Model judgments are not measured
accuracy against independent gold; shared OCR errors can fool extraction and
evaluation alike. Neither high scores nor passing gates approve a business
document. Gold curation, calibration, statistical certification, human review,
and deployment decisions remain application responsibilities.

## Local development setup

Clone the repository and synchronize declared development dependencies:

```powershell
git clone https://github.com/krprashantpk/extractly.git
Set-Location .\extractly
uv sync --all-groups
uv run python -c "import extractly; print(extractly.__version__)"
```

Use the root `.venv`; do not create another environment. In VS Code on Windows,
select `.venv\Scripts\python.exe`. The project is installed in editable mode,
so source changes do not require reinstalling. Read
[the coding conventions](../AGENTS.md) before contributing.

### Tests

```powershell
uv run pytest
uv run pytest -m "not sdk"
uv run pytest -m sdk
uv run ruff check .
uv run mypy src
```

The test suite uses synthetic fixtures and mocked service boundaries.
SDK-marked tests exercise actual local SDK execution without model requests.
No live grading or publishing is certified by these tests. CI runs Python
3.12, 3.13, and 3.14 on Ubuntu and Windows using
[the workflow](../.github/workflows/ci.yml).

### Build

```powershell
uv build
```

The `uv_build` backend creates wheel and source distributions under `dist/`.
Builds do not publish the package or deploy services.

## Notebooks

Local experiment notebooks cover offline, Foundry, and Decision exploration.
They are untracked exploratory application code, not
part of the installed API or prerequisites for the quick start. Review their
configuration before execution; some cells call remote services. Never commit
credentials, private fixtures, or sensitive outputs.

## Further reading

* [Architecture and ownership](../architecture.md)
* [Project structure](../Project-Structure.md)
* [Contributing](../CONTRIBUTING.md)
* [Azure AI Evaluation SDK](https://learn.microsoft.com/python/api/overview/azure/ai-evaluation-readme)
* [Microsoft Foundry evaluations](https://learn.microsoft.com/azure/ai-foundry/how-to/develop/evaluate-sdk?view=foundry-classic)
