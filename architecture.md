---
title: Extractly architecture
description: Component boundaries, evaluation flow, and resource ownership.
---

## Evaluation flow

```text
OfflineEvaluator                       OnlineEvaluator
CU dictionaries + required gold         CU dictionaries + source evidence
          |                                       |
OfflineEvaluationConfig                 OnlineEvaluationConfig
          |                                       |
CUAdapter + SchemaEvaluator              CUAdapter + SchemaEvaluator
          |                                       |
ExtractionEvaluator                     Optional semantic jobs:
Optional text F1                        groundedness, labels, completeness
          |                                       |
Field, table_row, table_count metrics     Reference-free assessment
          |                                       |
          +--- shared runner dispatch, coverage checks,
               score gates, and report provenance ---+
                                |
                    Report returned to the caller
```

## Responsibilities

* Shared configuration validates the extraction contract, schema, context budget,
  and gates. Offline configuration adds text diagnostics; online configuration
  adds semantic/model policy and grading budgets. Each rejects the other's
  workflow-specific settings.
* The adapter unwraps typed CU fields and retains evidence. It does not normalize
  values into application defaults.
* Evaluators validate canonical schema and compute occurrence-aware comparison.
  Arrays are compared only at corresponding positional indices, without keyed
  alignment. Configured array levels select `table_row_*` or `table_count_*`
  structural metrics independently of field-value correctness.
* The offline workflow requires gold, prepares reference-bearing datasets,
  validates extraction counts, and computes pooled comparison metrics.
* The online workflow prepares reference-free datasets, chooses semantic jobs,
  validates service decisions, and computes review recommendations.
* Execution runners implement `EvaluationRunner.run` with typed
  `RunnerRequest` and `RunnerResult` objects. Both workflows require
  explicit injection; `FoundryEvaluationRunner` delegates to the standard SDK
  evaluate function without an implicit fallback in either workflow.
* Private shared functions handle temporary dataset serialization, runner dispatch,
  generic output coverage, minimum gates, and report provenance. They do not
  choose a workflow or instantiate its evaluators.
* Applications own input loading, report persistence, authentication, and error
  handling. There is no command-line entry point.

Neither workflow imports, delegates to, or branches into the other. Their public
methods do not accept mode flags; supplying or omitting gold cannot switch
behavior. Offline single-document evaluation requires gold, and the online
single-document signature does not accept it.

## Ownership and failure contracts

Library callers configure authentication, provider policies, custom evaluators,
and required execution runner instances. The library never closes caller-owned
dependencies, commits transactions, creates cloud resources, or reads an
application's local settings.

Temporary JSONL inputs contain sensitive document evidence and are removed at
the end of the evaluation context, including on exceptions. This is filesystem
cleanup, not a secure-erasure guarantee. Native SDK behavior may create its own
telemetry or artifacts. Report retention belongs to the caller.

Failed or incomplete provider output raises rather than producing success-shaped
partial averages. Undefined positive-extraction metrics remain null. Field
comparators return one boolean; configured variation counts as a normal match
in every extraction score, with no separate tolerant score set. Run-level extraction means use
mean_document_precision, mean_document_recall, and mean_document_f1; pooled
occurrence-based scores use pooled_field_precision, pooled_field_recall, and
pooled_field_f1.
Individual document scores retain precision, recall, and f1.
Run-level extraction.exact_match is binary and requires all documents to match;
extraction.document_exact_match_rate is the fraction of matching documents.

## Deliberate boundaries

No document ingestion, persistence framework, retry engine, review workflow,
probability calibration, or business-specific validation is included. Semantic
grading and publishing are opt-in service operations. Neither evaluates the
correctness of the original OCR independently.
