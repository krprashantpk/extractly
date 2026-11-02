---
title: Extractly project structure
description: Source layout, public components, documentation, experiments, and validation tooling.
---

[README](README.md) | [User guide](docs/guide.md) |
[Architecture](architecture.md)

## Layout

The library uses a `src` layout. Documentation, tests, and exploratory notebooks
stay outside the importable package. Environments, caches, generated results,
and build distributions are omitted below.

```text
Extractly/
|-- .github/workflows/ci.yml
|-- AGENTS.md
|-- CONTRIBUTING.md
|-- LICENSE
|-- README.md
|-- architecture.md
|-- Project-Structure.md
|-- pyproject.toml
|-- uv.lock
|-- docs/
|   `-- guide.md
|-- src/extractly/
|   |-- __init__.py
|   |-- _shared.py
|   |-- comparison.py
|   |-- config.py
|   |-- cu/
|   |   |-- __init__.py
|   |   |-- _analyzer.py
|   |   |-- _pages.py
|   |   |-- adapter.py
|   |   |-- cu_result_parser.py
|   |   `-- ground_truth.py
|   |-- eval_runner/
|   |   |-- __init__.py
|   |   |-- base.py
|   |   |-- foundry.py
|   |   |-- local.py
|   |   `-- types.py
|   |-- evaluators/
|   |   |-- __init__.py
|   |   |-- _decision.py
|   |   |-- _online.py
|   |   |-- _shared.py
|   |   |-- extraction.py
|   |   |-- online_field.py
|   |   |-- online_structural.py
|   |   `-- schema.py
|   |-- offline.py
|   |-- online.py
|   |-- py.typed
|   `-- types.py
|-- tests/
`-- experiment/
```

Exploratory files under `experiment/` and generated `results/` may be local
workspace artifacts rather than versioned distribution content. Neither is
needed to run the documented synthetic examples.

## Core modules

* [Public API](src/extractly/__init__.py) exports separate online/offline
  workflows and configuration, the adapter, the ground-truth draft builder,
  deterministic evaluators, Decision evaluators, and typed runner contracts.
* [Configuration](src/extractly/config.py) defines strict Pydantic settings and
  accepts a single extraction model, table selectors, budgets, and gates.
* [Comparison models](src/extractly/comparison.py) own field selection, validation,
  and registered caller-defined comparisons through `EvaluationModel`.
* [CU adapter](src/extractly/cu/adapter.py) maps CU wire dictionaries to canonical
  extraction and evidence records.
* [Ground-truth drafts](src/extractly/cu/ground_truth.py) create editable
  canonical dictionaries for independent source review using the same
  [CU parser](src/extractly/cu/cu_result_parser.py) as the adapter.
* [Analyzer interpretation](src/extractly/cu/_analyzer.py) and
  [page preparation](src/extractly/cu/_pages.py) retain original analyzer
  instructions, physical-page evidence, record paths, and provenance deferrals.
* [Evaluators](src/extractly/evaluators/__init__.py) expose separate modules for
  [schema checks](src/extractly/evaluators/schema.py) and
  [occurrence-aware extraction metrics](src/extractly/evaluators/extraction.py),
  plus [field judgments](src/extractly/evaluators/online_field.py) and
  [structural judgments](src/extractly/evaluators/online_structural.py).
* [Execution runners](src/extractly/eval_runner/base.py) define an abstract
  execution boundary with [typed requests and results](src/extractly/eval_runner/types.py).
  The [local runner](src/extractly/eval_runner/local.py) invokes synchronous
  callables; the [Foundry runner](src/extractly/eval_runner/foundry.py) uses the
  SDK and optional caller-configured publishing. Both workflows require an
  explicit runner instance.
* [Offline evaluation](src/extractly/offline.py) owns gold-based configuration,
  preparation, scoring, and pooled occurrence metrics.
* [Online evaluation](src/extractly/online.py) owns reference-free configuration,
  preparation, injected Decision checks, page progress, and coverage reports.
* [Shared mechanics](src/extractly/_shared.py) handles dataset serialization,
  runner dispatch/coverage checks, gates, and compact reports without a
  mode-switching orchestrator.
* [Shared types](src/extractly/types.py) isolate open-ended JSON/provider boundary
  types. The package includes a PEP 561 marker.

## Documentation and examples

| Location | Purpose |
| --- | --- |
| [README](README.md) | Overview, installation, and self-contained local quick starts |
| [User guide](docs/guide.md) | Workflow examples, comparison policy, components, runners, metrics, and development |
| [Architecture](architecture.md) | Evaluation flow, ownership boundaries, publishing, and failure contracts |
| [Contributing](CONTRIBUTING.md) | Contribution scope, synthetic tests, and validation commands |
| [Repository conventions](AGENTS.md) | Python design, dependency, documentation, and validation conventions |
| Local experiments (untracked) | Offline, Foundry, and Decision exploration outside the installed library |

The guide's examples do not depend on a separate samples directory. Review
notebook configuration before execution; exploratory notebooks may make
billable remote requests or produce sensitive outputs.

## Tooling

[Project metadata](pyproject.toml) declares Python compatibility, dependencies,
the uv build backend, and lint/type/test configuration. The root virtual
environment and lockfile are managed with uv.

[Tests](tests/) use synthetic inputs and include local SDK execution plus mocked
semantic boundaries. [CI](.github/workflows/ci.yml) validates supported Python
versions (3.12, 3.13, and 3.14) on Windows and Ubuntu, checks lint and types,
runs tests, and builds distributions. It does not publish packages or deploy
Azure. See [development setup](docs/guide.md#local-development-setup) for commands.
