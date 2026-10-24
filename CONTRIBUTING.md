---
title: Contributing to Extractly
description: Local development, validation, and contribution boundaries.
---

## Set up development

Use Python 3.12 or newer and the root virtual environment:

```powershell
uv sync --all-groups
uv run pytest
uv run ruff check .
uv run mypy src\extractly
uv build
```

## Keep contributions reusable

* Follow [the repository guidelines](AGENTS.md).
* Accept caller-configured dependencies and keep ownership explicit.
* Add synthetic regression tests for behavioral changes.
* Preserve missing-value, duplicate-occurrence, and evidence semantics.
* Keep model grading optional and never disguise provider failure as success.
* Keep online and offline orchestration and configuration separate. Share narrow
  components rather than adding a mode-switching workflow or package CLI.
* Do not contribute credentials, private documents, customer configurations,
  production outputs, or code you are not authorized to redistribute.
* Use the [sample](samples/document-evaluation/README.md) for application examples
  rather than adding application business rules to the package.

## Validate changes

Deterministic tests must run without credentials. SDK-marked tests use Foundry's
local execution; model evaluation and publishing require separately authorized
resources. Include behavior and compatibility notes when changing configuration
or report formats.

Contributions are provided under [the MIT license](LICENSE).
