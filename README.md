---
title: Extractly
description: Evidence-preserving evaluation for Azure Content Understanding extraction.
---

## Extractly

Document extraction evaluation for Python. Compare fields, assess evidence,
and report what needs review.

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-29814A)](LICENSE)

[Installation](#installation) | [Quick start](#quick-start) |
[Documentation](#documentation) | [Contributing](#contributing)

Extractly evaluates saved Azure Content Understanding (CU) results. Compare
predictions against independent gold, or assess schema and source evidence
without gold. Choose an explicit local or Microsoft Foundry execution runner
and keep clients, credentials, and report storage in your application.

```text
Saved CU result + independent gold --> OfflineEvaluator --> Comparison metrics
Saved CU result + source evidence  --> OnlineEvaluator  --> Schema / Decision checks
                                                               |
                                           Metrics + gates + document review flags
```

## Features

* Typed field selection and validation through Pydantic-based `EvaluationModel`
* Caller-defined comparison tolerances through `@field_comparator`
* Field precision, recall, F1, exact match, and positional row/table metrics
* Reference-free schema checks and optional page-based Decision judgments
* Preserved spans, confidence, Markdown, missing fields, and duplicate occurrences
* Explicit execution runners, minimum-score gates, and configuration fingerprints
* Optional Foundry publishing of precomputed per-document scores

## Installation

Requires Python 3.12 or newer. From an existing
[uv](https://docs.astral.sh/uv/getting-started/installation/)-managed project:

```powershell
uv add "extractly @ git+https://github.com/krprashantpk/extractly.git"
```

Git must be installed. Append `@<tag-or-commit>` to pin a source revision, or
use `uv add C:\path\to\Extractly` for a local checkout. The package has not
been published to PyPI. For library development, use
[the development setup](#local-development).

## Quick start

Evaluate a synthetic CU result without Azure credentials or external files:

```python
"""Compare a synthetic invoice with independently written gold."""

from decimal import Decimal

from extractly import EvaluationModel, OfflineEvaluationConfig, OfflineEvaluator
from extractly.eval_runner import LocalRunner


class Invoice(EvaluationModel):
    """Select the total for validation and comparison."""

    total: Decimal


raw = {
    "contents": [{
        "category": "Invoice",
        "startPageNumber": 1,
        "endPageNumber": 1,
        "fields": {"total": {"type": "number", "valueNumber": 125.5}},
    }],
}
gold = {
    "segments": [{
        "category": "Invoice",
        "pages": [1, 1],
        "fields": {"total": 125.5},
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
```

`extraction_model` describes each segment's `fields`, not the complete CU
response. Declared fields are scored; undeclared fields remain unscored.
For real documents, gold must be independently verified against the source.

See the [user guide](docs/guide.md#usage) for batches, saved input, custom
comparison rules, and report persistence.

### Evaluate without gold

Reuse `Invoice` and `raw` from the preceding example for deterministic schema
checks:

```python
from extractly import OnlineEvaluationConfig, OnlineEvaluator

online = OnlineEvaluator(OnlineEvaluationConfig(extraction_model=Invoice), runner=LocalRunner())
online_report = online.evaluate(raw, document_id="invoice-001")
assert online_report["metrics"]["schema.valid"] == 1
```

Schema validity does not prove source support or completeness. To assess
fields and structure against page evidence, inject caller-configured
`OnlineFieldEvaluator` and `OnlineStructuralEvaluator` instances with the
original analyzer definition. See
[Decision evaluation](docs/guide.md#decision-field-and-structural-assessment).

## Integrations

| Responsibility | Built-in support |
| --- | --- |
| Extraction input | Saved Azure Content Understanding response dictionaries |
| Local execution | `LocalRunner` with synchronous evaluator callables |
| SDK execution | `FoundryEvaluationRunner` with Azure AI Evaluation |
| Semantic judgments | TypeSafe Decision client supplied by the caller |
| Cloud publishing | Optional classic Foundry project or caller-owned OpenAI client |

Extractly does not run CU analysis or provision services. Remote judging and
publishing require authorized resources and data-owner approval. A passing
gate or high model score is not automatic approval of a business document.

## Extending Extractly

* Declare an `EvaluationModel` and register field comparison rules
* Supply application-owned evaluator callables through `evaluators=`
* Subclass `EvaluationRunner` to implement another execution backend
* Use `CUAdapter`, `SchemaEvaluator`, or `ExtractionEvaluator` independently

No plugin registration is required. The
[extension guide](docs/guide.md#extending-extractly) documents signatures,
output contracts, and dependency ownership.

## Documentation

* [User guide and component reference](docs/guide.md)
* [Comparison models and table metrics](docs/guide.md#comparison-models)
* [Online schema and Decision evaluation](docs/guide.md#online-evaluation)
* [Execution runners and Foundry publishing](docs/guide.md#execution-runners)
* [Reports, metrics, and score gates](docs/guide.md#reports-and-metrics)
* [Architecture and design](architecture.md)
* [Project structure](Project-Structure.md)
* [Local development and testing](docs/guide.md#local-development-setup)

## Local development

Clone the repository and synchronize the root environment:

```powershell
git clone https://github.com/krprashantpk/extractly.git
Set-Location .\extractly
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy src
uv build
```

Tests use synthetic inputs and mocked semantic boundaries. SDK-marked tests
exercise local Foundry SDK execution without model requests. Live judging and
publishing are not certified by the local suite. Use the root `.venv` for all
development commands.

## Contributing

Bug reports, documentation improvements, examples, and integrations are
welcome. [Open an issue](https://github.com/krprashantpk/extractly/issues) to
report a problem or discuss a larger change. Follow the
[contribution guide](CONTRIBUTING.md) and [repository conventions](AGENTS.md),
and include synthetic tests for behavior changes.

Extractly is licensed under [MIT](LICENSE).
