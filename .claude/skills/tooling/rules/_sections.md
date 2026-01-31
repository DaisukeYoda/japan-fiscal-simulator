# Section Definitions

## Analysis (analysis)
**Impact:** HIGH

Structural code analysis for quality assessment.
Detects dead code, duplicate code, circular dependencies, and excessive complexity.

**Rules:**
- `analysis-pyscn` - Dead code, clones, dependencies, complexity with pyscn

## Linting (lint)
**Impact:** CRITICAL

Static code analysis with ruff for consistent, high-quality code.
Ruff is an extremely fast Python linter written in Rust, replacing flake8, isort, and many other tools.

**Rules:**
- `lint-ruff` - Fast, comprehensive linting with ruff

## Type Checking (type)
**Impact:** HIGH

Static type checking with mypy for type safety and better IDE support.
Catches type errors at development time, improving code reliability and maintainability.

**Rules:**
- `type-mypy` - Static type checking with mypy

## Formatting (fmt)
**Impact:** HIGH

Consistent code formatting with ruff format and import sorting.
Eliminates style debates and ensures consistent code appearance across the project.

**Rules:**
- `fmt-ruff` - Code formatting and import sorting with ruff

## Testing (test)
**Impact:** HIGH

Test framework configuration with pytest for reliable testing.
Covers configuration, fixtures, parametrization, and coverage settings.

**Rules:**
- `test-pytest` - Testing with pytest, fixtures, and coverage

## Package Management (pkg)
**Impact:** MEDIUM

Modern Python packaging with uv and pyproject.toml.
Fast, reliable dependency management and project configuration.

**Rules:**
- `pkg-uv` - Fast package management with uv
- `pkg-pyproject` - Project configuration with pyproject.toml
