---
title: Repository Coding Guidelines
description: Project-agnostic Python coding conventions and design principles.
---

## Design Principles

### Separation Of Responsibilities

- Give each component a focused responsibility and define clear interfaces between components.
- Keep orchestration separate from implementation details. Coordinating components should delegate work rather than contain dependency-specific or persistence logic.
- Reuse existing interfaces and shared models when adding implementations. Add abstractions only when they serve a concrete responsibility or interchangeable implementation.

### Caller-Owned Dependencies

- Accept caller-configured clients and collaborators through explicit dependency injection, preferring constructor injection for class dependencies.
- Leave client creation, authentication, retries, timeouts, and cleanup to the caller or supplied client. Do not configure them implicitly inside consuming components.
- Keep resource and transaction ownership explicit. Do not close caller-owned clients or commit caller-owned transactions.
- Expose optional behavior through explicit parameters rather than hidden configuration or global state.

## Python Environment And Dependencies

- Use `uv` for Python package and environment management. Do not invoke `pip`, Poetry, Pipenv, or Conda directly.
- Reuse the target repository's root `.venv/`; do not create another virtual environment.
- Add project dependencies with `uv add <package>` or `uv add --group <group> <package>` so `pyproject.toml` remains authoritative.
- Install an intentionally untracked, non-project package with `uv pip install <package>`.
- Synchronize declared dependencies with `uv sync` or `uv sync --all-groups`.
- Run Python tools through the project environment with `uv run <command>` or the root virtual environment's Python executable.

## Python Documentation And Types

- Add Google-style docstrings to every Python module, class, function, and method, including constructors and private methods where meaningful.
- Keep the opening summary short. Use `Args:`, `Returns:`, `Yields:`, `Raises:`, `Attributes:`, and `Examples:` only when applicable.
- Keep each docstring section entry on one physical line. Write descriptions as prose without backticks, snippets, or cosmetic line wrapping.
- Add complete type annotations to every parameter and return value, including `-> None` where appropriate.
- Prefer Python 3.10+ syntax such as `list[str]` and `X | None`. Use precise collection abstractions from `collections.abc` for parameters and concrete types for returns.
- Avoid `Any` unless unavoidable, and document why it is required.
- Use descriptive names and keyword-only parameters for optional settings whose meaning would be unclear in positional calls.
- Use dataclasses for shared structured results, preferring `frozen=True` when fields should not be reassigned.
- Document observable guarantees, relevant limits, and ownership boundaries in public API docstrings.

## Method Signatures And Calls

- Keep method signatures and method calls on one physical line unless the parameter or argument list contains more than six items.

## Logging

- Use the standard-library `logging` module and declare `logger = logging.getLogger(__name__)` at module level.
- Use lazy `%` formatting with positional arguments. Do not format log messages with f-strings or `.format()`.
- Keep each logger call on one physical line and quote interpolated string values with single quotes in the message.
- Use `info` for normal progress and milestones, `warning` for recoverable degradation, `error` for handled failures, and `exception` inside exception handlers when a traceback is useful.
- Pair operation start and success messages. Include relevant identifiers and quantitative results so concurrent work remains traceable.

## Exceptions

- Validate configuration during construction and operation inputs before processing or external calls. Validate dependency results before using them in subsequent operations.
- Raise `TypeError` for unsupported input types and `ValueError` for invalid values. Reject booleans explicitly when an integer-only setting requires it.
- Use clear error messages that identify the invalid parameter or violated contract. Raise `RuntimeError` when a provider result violates an expected contract.
- Let provider exceptions propagate unchanged unless a documented API contract requires translation. Do not silently suppress failures or substitute success-shaped results.
- Keep each `raise` statement on one physical line, including the exception constructor, message, and any `from` clause.
