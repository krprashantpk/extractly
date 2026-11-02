---
title: Contributing to Extractly
description: Local development, validation, and contribution boundaries.
---

[README](README.md) | [User guide](docs/guide.md) |
[Architecture](architecture.md)

## Set up development

Use Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).
From the repository root, synchronize declared dependencies:

```powershell
uv sync --all-groups
```

Use the root `.venv` for every command. In VS Code on Windows, select
`.venv\Scripts\python.exe`. Source changes are available through the editable
installation. No Azure resource or local server is needed for deterministic
examples and mocked tests. See [development setup](docs/guide.md#local-development-setup)
for the complete setup and test categories.

## Propose a change

[Open an issue](https://github.com/krprashantpk/extractly/issues) for a bug report
or to discuss a larger behavior or integration change. Include a minimal
synthetic reproduction, expected behavior, actual behavior, and relevant
versions. Remove credentials, private source text, and identifying report data.

## Keep contributions reusable

* Follow [the repository guidelines](AGENTS.md).
* Accept caller-configured dependencies and keep ownership explicit.
* Add synthetic regression tests for behavioral changes.
* Preserve missing-value, duplicate-occurrence, and evidence semantics.
* Keep model grading optional and never disguise provider failure as success.
* Keep online and offline orchestration and configuration separate. Share narrow
  components rather than adding a mode-switching workflow or package CLI.
* Require an explicit runner and preserve caller ownership of clients,
  credentials, evaluator instances, retries, and cleanup.
* Do not contribute credentials, private documents, customer configurations,
  production outputs, or code you are not authorized to redistribute.
* Add reusable examples to the [user guide](docs/guide.md) rather than putting
  application business rules in the package.

## Validate changes

Run the narrow tests for your change while iterating, then the required gates:

```powershell
uv run pytest
uv run ruff check .
uv run mypy src
uv build
```

Deterministic tests must run without credentials. SDK-marked tests use actual
local SDK execution without model requests. Decision tests mock the client;
Foundry publishing tests mock the cloud boundary. Live grading and publishing
require separately authorized resources and are not certified by the local
suite.

For new runners, test typed input/output, exact ID coverage, invalid results,
and provider failures. For comparison or page checks, cover missing values,
duplicates, positional arrays, incomplete evidence, and unknown metrics when
those behaviors are affected. Include compatibility notes for configuration,
public signatures, and report changes.

## Update documentation

Keep the [README](README.md) focused on installation and a runnable local
example. Put detailed workflow examples and component contracts in the
[user guide](docs/guide.md). Update the
[architecture](architecture.md) or [project structure](Project-Structure.md)
when responsibilities or modules change.

Use Markdown frontmatter with `title` and `description`, explicit code-fence
languages, and valid relative links. Execute self-contained examples, compile
Python snippets, and clearly identify examples that need caller-supplied
files, clients, or remote services. Never use live credentials to validate a
documentation-only change.

## Submit a pull request

Describe the behavior or documentation change, explain relevant tradeoffs,
and list the checks run. Keep unrelated refactors out of the change and use
synthetic fixtures for added tests. CI checks Python 3.12 through 3.14 on
Windows and Ubuntu; local success does not replace those compatibility gates.

Contributions are provided under [the MIT license](LICENSE).
