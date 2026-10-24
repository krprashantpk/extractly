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
Pooled field/row/instance metrics         Reference-free assessment
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
  Matching uses independent identities rather than quantities being graded.
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
partial averages. Undefined positive-extraction metrics remain null. Strict and
tolerant field scores are distinct. The SDK's per-document means remain separate
from pooled occurrence-based micro metrics.

## Deliberate boundaries

No document ingestion, persistence framework, retry engine, review workflow,
probability calibration, or business-specific validation is included. Semantic
grading and publishing are opt-in service operations. Neither evaluates the
correctness of the original OCR independently.
