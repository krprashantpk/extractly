---
title: Extractly project structure
description: Source layout, public modules, samples, and validation tooling.
---

## Layout

The library follows a source layout with application examples outside the
importable package. Generated environments, caches, and distributions are omitted.

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
|-- src/extractly/
|   |-- __init__.py
|   |-- _shared.py
|   |-- comparison.py
|   |-- config.py
|   |-- cu/
|   |   |-- __init__.py
|   |   |-- adapter.py
|   |   |-- cu_result_parser.py
|   |   `-- ground_truth.py
|   |-- eval_runner/
|   |   |-- __init__.py
|   |   |-- base.py
|   |   |-- foundry.py
|   |   `-- types.py
|   |-- evaluators.py
|   |-- offline.py
|   |-- online.py
|   |-- py.typed
|   `-- types.py
|-- tests/
`-- samples/document-evaluation/
    |-- README.md
    |-- offline.json
    |-- online.json
    |-- offline.py
    |-- online.py
    `-- synthetic.jsonl
```

## Core modules

* [Public API](src/extractly/__init__.py) exports separate online/offline
  workflows and configuration, the adapter, the ground-truth draft builder,
  and deterministic evaluators.
* [Configuration](src/extractly/config.py) defines strict Pydantic settings and
  accepts a single extraction schema model.
* [Comparison models](src/extractly/comparison.py) own field selection, validation,
  and registered caller-defined comparisons through `EvaluationModel`.
* [CU adapter](src/extractly/cu/adapter.py) maps CU wire dictionaries to canonical
  extraction and evidence records.
* [Ground-truth drafts](src/extractly/cu/ground_truth.py) create editable
  canonical dictionaries for independent source review using the same
  [CU parser](src/extractly/cu/cu_result_parser.py) as the adapter.
* [Evaluators](src/extractly/evaluators.py) implement schema checks and typed,
  occurrence-aware extraction metrics.
* [Execution runners](src/extractly/eval_runner/base.py) define an abstract
  execution boundary with [typed requests and results](src/extractly/eval_runner/types.py).
  The [Foundry runner](src/extractly/eval_runner/foundry.py) calls the standard
  SDK evaluate function. Both workflows require an explicit runner instance.
* [Offline evaluation](src/extractly/offline.py) owns gold-based configuration,
  preparation, scoring, and pooled occurrence metrics.
* [Online evaluation](src/extractly/online.py) owns reference-free configuration,
  preparation, semantic graders, and assessment decisions.
* [Shared mechanics](src/extractly/_shared.py) handles dataset serialization,
  runner dispatch/coverage checks, gates, and report provenance without a
  mode-switching orchestrator.
* [Shared types](src/extractly/types.py) isolate open-ended JSON/provider boundary
  types. The package includes a PEP 561 marker.

## Tooling and samples

[Project metadata](pyproject.toml) declares Python compatibility, dependencies,
the uv build backend, and lint/type/test configuration. The root virtual
environment and lockfile are managed with uv.

[Tests](tests/) use synthetic inputs and include local SDK execution plus mocked
semantic boundaries. [CI](.github/workflows/ci.yml) validates supported Python
versions and builds distributions. It does not publish packages or deploy Azure.

[The document-evaluation sample](samples/document-evaluation/README.md)
demonstrates explicit configuration without any customer application imports.
