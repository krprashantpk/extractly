---
title: Extractly project structure
description: Evaluator families, pipeline modules, shared CU parsing, and validation tooling.
---

[README](README.md) | [User guide](docs/guide.md) |
[Architecture](architecture.md)

## Layout

```text
src/extractly/
|-- __init__.py
|-- _pipeline.py
|-- comparison.py
|-- config.py
|-- offline_pipeline.py
|-- online_pipeline.py
|-- types.py
|-- py.typed
|-- cu/
|   |-- __init__.py
|   `-- cu_result_parser.py
|-- eval_runner/
|   |-- __init__.py
|   |-- base.py
|   |-- foundry.py
|   |-- local.py
|   `-- types.py
|-- store/
|   |-- __init__.py
|   |-- base.py
|   `-- blob.py
`-- evaluators/
    |-- __init__.py
    |-- _shared.py
    |-- field.py
    |-- structural.py
    |-- llm_field.py
    |-- llm_structural.py
    `-- schema.py
```

Documentation, tests, and exploratory notebooks stay outside the importable
package. Existing experiments and generated results are not migrated with
public API changes.

## Public components

* [Evaluators](src/extractly/evaluators/__init__.py) expose deterministic and
  LLM field/structural evaluation plus schema validation.
* [Offline pipeline](src/extractly/offline_pipeline.py) defaults to field,
  structural, and schema evaluation against independent references.
* [Online pipeline](src/extractly/online_pipeline.py) defaults to the two LLM
  evaluators and schema validation with a caller-owned client and deployment,
  accepting one file per call and reporting direct metrics with a review flag.
* [Pipeline mechanics](src/extractly/_pipeline.py) bind evaluator parameters,
  execute runner jobs, validate coverage, and build separate offline aggregate
  and single-document online metric reports.
* [Comparison models](src/extractly/comparison.py) retain typed field selection,
  constraints, aliases, and custom static comparison policies.
* [CU parser](src/extractly/cu/cu_result_parser.py) provides the single
  conversion path used by both pipelines and standalone callers. Offline
  references are supplied independently rather than generated from predictions.
* [Execution runners](src/extractly/eval_runner/base.py) retain explicit local
  or SDK execution and caller-owned publishing dependencies.
* [Golden-dataset stores](src/extractly/store/base.py) load independent
  ground-truth objects keyed by document ID. The
  [Azure Blob implementation](src/extractly/store/blob.py) reads one JSON blob
  per document beneath a prefix using a caller-owned container client.

## Validation

[Project metadata](pyproject.toml) defines Python compatibility, dependencies,
pytest, Ruff, mypy, and the uv build backend. Use the root `.venv` for commands
listed in [the contribution guide](CONTRIBUTING.md).

Tests cover CU transformation, comparator registration, standalone evaluators,
pipeline defaults, custom metric contracts, occurrence aggregation,
reference-free inputs, and runner ownership. Provider calls are mocked; live
inference and publishing require separate authorization and verification.