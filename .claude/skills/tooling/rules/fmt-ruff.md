---
title: Formatting with Ruff
impact: HIGH
impactDescription: Consistent code style
tags: [ruff, formatting, black, isort]
---

# Formatting with Ruff [HIGH]

## Description
Ruff's formatter is a drop-in replacement for Black, written in Rust for extreme speed. Combined with isort rules, it handles both code formatting and import organization in one tool.

## Basic Configuration

```toml
# pyproject.toml
[tool.ruff]
line-length = 88

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"

exclude = [
    "migrations/",
    "*.pyi",
]
```

## Import Sorting

```toml
[tool.ruff.lint]
select = ["I"]  # Enable isort rules

[tool.ruff.lint.isort]
known-first-party = ["myproject"]
known-third-party = ["fastapi", "pydantic"]
force-single-line = false
lines-after-imports = 2
section-order = [
    "future",
    "standard-library",
    "third-party",
    "first-party",
    "local-folder",
]
```

## Import Order

```python
# Properly sorted imports
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from pydantic import BaseModel

from myproject.core import service
from myproject.utils import helper

if TYPE_CHECKING:
    from myproject.models import User
```

## Line Length

```toml
# Consistent across all tools
[tool.ruff]
line-length = 88

[tool.ruff.lint.pycodestyle]
max-line-length = 88
```

| Length | Rationale |
|--------|-----------|
| 79 | PEP 8 original |
| 88 | Black/Ruff default (recommended) |
| 100 | More horizontal space |
| 120 | Wide monitors |

## Commands

```bash
ruff format .              # Format all files
ruff format . --check      # Check without modifying
ruff format . --diff       # Show diff
ruff check . --select=I --fix  # Fix import order
```

## Notes
- Ruff format is compatible with Black's style
- Use the same `line-length` for linting and formatting
- `known-first-party` is usually auto-detected but explicit is better
- No configuration is often best - embrace the defaults

## References
- [Ruff Formatter](https://docs.astral.sh/ruff/formatter/)
- [Ruff isort Rules](https://docs.astral.sh/ruff/rules/#isort-i)
