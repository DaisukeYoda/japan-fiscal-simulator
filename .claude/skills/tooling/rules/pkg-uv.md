---
title: Package Management with uv
impact: MEDIUM
impactDescription: 10-100x faster dependency management
tags: [uv, package-management, pip, virtualenv, lock-file]
---

# Package Management with uv [MEDIUM]

## Description
uv is an extremely fast Python package manager written in Rust by Astral. It's a drop-in replacement for pip, pip-tools, and virtualenv that's 10-100x faster. Use it for virtualenv creation, package installation, and lock file generation.

## Basic Usage

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtualenv (instant)
uv venv

# Install from pyproject.toml
uv pip install -e ".[dev]"

# Install specific packages
uv pip install fastapi pydantic

# Install from requirements.txt
uv pip install -r requirements.txt

# Run without installing
uvx ruff check .
uvx pytest
```

## Lock Files

```bash
# Generate lock file from pyproject.toml
uv pip compile pyproject.toml -o requirements.lock

# With hashes for security
uv pip compile pyproject.toml -o requirements.lock --generate-hashes

# Update all dependencies
uv pip compile pyproject.toml -o requirements.lock --upgrade

# Update specific package
uv pip compile pyproject.toml -o requirements.lock --upgrade-package fastapi

# Install exact versions from lock file
uv pip sync requirements.lock
```

## Project Structure

```
project/
├── pyproject.toml         # Source of truth (loose versions)
├── requirements.lock      # Production lock file
└── requirements-dev.lock  # Development lock file
```

```bash
# Generate both lock files
uv pip compile pyproject.toml -o requirements.lock
uv pip compile pyproject.toml --extra dev -o requirements-dev.lock
```

## Commands

| Command | Purpose |
|---------|---------|
| `uv venv` | Create virtualenv |
| `uv pip install` | Install packages |
| `uv pip sync` | Sync to requirements exactly |
| `uv pip compile` | Generate lock file |
| `uv pip list` | List installed packages |
| `uv run` | Run command in environment |
| `uvx` | Run tool without installing |

## Workflow

| Environment | Command |
|-------------|---------|
| Development | `uv pip sync requirements-dev.lock` |
| CI | `uv pip sync requirements-dev.lock` |
| Production | `uv pip sync requirements.lock` |

## Notes
- uv uses a global cache, making repeated installs instant
- Lock files should be committed to version control
- Use `uv pip sync` (not install) for exact reproduction
- `uvx` is great for one-off tool execution
- Update lock files regularly for security patches

## References
- [uv Documentation](https://docs.astral.sh/uv/)
- [uv GitHub](https://github.com/astral-sh/uv)
