## Extractly

Document extraction evaluation for Python. Compare fields, validate structure, and check schemas.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)

[Installation](#installation) | [Quick start](#quick-start) | [Documentation](#documentation) | [Contributing](#contributing)

Extractly evaluates saved Azure Content Understanding (CU) results against
independently reviewed ground truth or source Markdown. Compose an offline or
online pipeline from interchangeable evaluators and an execution runner, or use
each evaluator on its own.

```text
CU result -> Parsed segments -> Evaluators -> Metrics and reports
```

## Features

* Deterministic field and table-structure evaluation against independent ground truth
* Source-grounded LLM judgments using Markdown and the original analyzer definition
* Independent schema validation for extracted types and constraints
* Typed comparison models, custom comparison policies, and replaceable evaluators
* Local and Microsoft Foundry execution runners with caller-owned dependencies
* Offline aggregate metrics and minimum-score gates; online per-document review signals

## Installation

Requires Python 3.12 or newer. Install from Git in an existing
[uv](https://docs.astral.sh/uv/getting-started/installation/)-managed project:

```powershell
uv add "extractly @ git+https://github.com/krprashantpk/extractly.git"
```

Git must be installed. You can pin the dependency by appending `@<tag-or-commit>` to the URL.

## Quick start

Already have a CU result? Evaluate it against independent ground truth without
credentials or network calls:

```python
"""Evaluate a synthetic delivery with the three default offline checks."""

from extractly import EvaluationModel, OfflineEvaluationConfig, OfflineEvaluationPipeline
from extractly.eval_runner import LocalRunner


class LineItem(EvaluationModel):
    """Select a quantity in each table row."""

    quantity: int


class Delivery(EvaluationModel):
    """Select the delivery's row table."""

    items: list[LineItem]


raw = {
    "contents": [{
        "markdown": "Delivery: quantity 10",
        "fields": {"items": {"valueArray": [
            {"valueObject": {"quantity": {"valueInteger": 10}}},
        ]}},
    }],
}
gold = {"items": [{"quantity": 10}]}
config = OfflineEvaluationConfig(extraction_model=Delivery, thresholds={"field.extraction.pooled_field_f1": 0.95})
pipeline = OfflineEvaluationPipeline(config, runner=LocalRunner())
documents = [{"id": "delivery-001", "cu_result": raw, "ground_truth": gold}]
report = pipeline.run(documents)

assert report["metrics"]["field"]["extraction.pooled_field_f1"] == 1.0
assert report["metrics"]["structural"]["extraction.table_row.tp"] == 1
assert report["metrics"]["schema"]["valid"] == 1
assert report["gate_status"] == "pass"
```

Pass one or more document records, each containing `id`, `cu_result`, and
`ground_truth`. The default offline pipeline runs `field`, `structural`, and
`schema` checks. The report groups metrics by evaluator name and includes
per-document metrics, document means, pooled occurrence scores, and configured
gates. Means and pooled scores are distinct.

Record-array tables come from the extraction model, not table rules. For
scalar-only models, explicitly select field and schema evaluators without the
structural evaluator.

See the [offline evaluation guide](docs/guide.md#offline-evaluation) for dataset
evaluation and [Azure Blob Storage examples](docs/guide.md#golden-datasets-in-azure-blob-storage)
for loading independent gold annotations.

### Online evaluation

`OnlineEvaluationPipeline` defaults to LLM field, LLM structural, and schema
checks. Pass your configured synchronous OpenAI `client` and `deployment` to the
pipeline, and the original analyzer definition with each document. These defaults
make remote inference requests, even when using `LocalRunner`.

To run schema-only assessment without a model service, reuse `Delivery` and
`raw` from the quick start:

```python
from extractly import OnlineEvaluationConfig, OnlineEvaluationPipeline, SchemaEvaluator

online_config = OnlineEvaluationConfig(extraction_model=Delivery)
online = OnlineEvaluationPipeline(online_config, runner=LocalRunner(), evaluators={"schema": SchemaEvaluator()})
online_report = online.run(raw, document_id="delivery-001")
assert online_report["metrics"]["schema"]["valid"] == 1
assert online_report["review_required"] is False
```

Online `run(...)` and its `evaluate(...)` alias accept exactly one CU result
dictionary, not a batch. Ground-truth labels never reach online evaluators.
Reports contain direct document metrics, `id`, and `review_required`, not
offline document lists, means, pooled scores, or gates.

Online thresholds use document metric keys such as `field.extraction.document.f1`.
Thresholds are checked before rounding; missing or undefined scores require
review. See the [online evaluation example](docs/guide.md#online-evaluation)
for client setup requirements and the [report reference](docs/guide.md#reports-and-metrics)
for review signals and metric contracts.

## Evaluators

| Evaluator | Evidence | Responsibility |
| --- | --- | --- |
| `FieldEvaluator` | Independent gold | Typed values, registered comparisons, field TP/FP/FN |
| `StructuralEvaluator` | Independent gold | Model-derived tables and positional row presence |
| `LLMFieldEvaluator` | Source Markdown and analyzer | Source-grounded field judgments |
| `LLMStructuralEvaluator` | Source Markdown and analyzer | Source counts and structural integrity judgments |
| `SchemaEvaluator` | Extraction model | Candidate type and constraint validation |

Evaluators receive normalized field dictionaries and the extraction model in
their call parameters, not configuration objects. LLM evaluators additionally
retain a caller-owned OpenAI client and deployment. Matching table and row counts
alone do not establish row identity, ordering, or field correctness.

The [standalone evaluator reference](docs/guide.md#standalone-evaluators) covers
call signatures, evidence requirements, metrics, and diagnostics.

## Extending Extractly

Pass an explicit `evaluators=` mapping to either pipeline to replace its
defaults. Custom evaluators are synchronous callables that receive their declared
subset of normalized call parameters and return `{"metrics": {...}}`. Metric
values must be finite numbers, booleans, or `None`; separate diagnostics are not
retained in pipeline reports.

Use `EvaluationModel` and `@field_comparator` for custom static field-comparison
policies, or implement `EvaluationRunner` to change execution. Both pipelines
require an explicit `runner=`. Clients, credentials, retries, timeouts, cleanup,
and report persistence remain caller-owned; the pipelines do not create or close
clients.

See the [extension guide](docs/guide.md#extending-extractly),
[comparison models](docs/guide.md#comparison-models), and
[execution runners](docs/guide.md#execution-runners) for contracts and examples.

## CU parsing

Both pipelines use `parse_cu_result` to normalize saved CU results. It returns a
`ParsedCUResult` containing typed `ParsedCUSegment` instances, available from
`extractly` or `extractly.cu`. Access `segment.markdown` and `segment.fields`
directly.

Confidence, geometry, and span evidence are not collected. Each evaluator receives
only its segment's source Markdown. Offline ground truth must be independently
reviewed, not generated from the predictions being evaluated. See the
[parsing guide](docs/guide.md#cu-parsing) for segment and evidence boundaries.

## Documentation

* [User guide and evaluator reference](docs/guide.md)
* [Architecture and design](architecture.md)
* [Project structure](Project-Structure.md)
* [Local development and testing](docs/guide.md#local-development-setup)
* [Migration from removed evaluators and CU helpers](docs/guide.md#migration)

Existing exploratory notebooks may use removed APIs and require migration before execution.

## Local development

Clone the repository and synchronize its development dependencies:

```powershell
git clone https://github.com/krprashantpk/extractly.git
Set-Location .\extractly
uv sync --all-groups
```

Run tests, lint, type-check, and build from the repository root:

```powershell
uv run pytest
uv run ruff check .
uv run mypy src
uv build
```

Use the root `.venv` for development. Local tests use synthetic documents and
mocked provider boundaries. Remote judging and publishing need authorized
resources and data-owner approval. Model scores, review flags, and passing gates
are not automatic business approval.

## Contributing

Bug reports, documentation improvements, examples, and new evaluators are welcome.
[Open an issue](https://github.com/krprashantpk/extractly/issues) to report a bug
or discuss a larger change before starting work.

Include tests for behavior changes and follow the
[repository coding guidelines](AGENTS.md). See the
[contributing guide](CONTRIBUTING.md) for development and submission requirements.

Extractly is licensed under [MIT](LICENSE).