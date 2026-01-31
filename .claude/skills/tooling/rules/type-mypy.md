---
title: Type Checking with Mypy
impact: HIGH
impactDescription: Static type safety
tags: [mypy, type-checking, typing, pyproject.toml]
---

# Type Checking with Mypy [HIGH]

## Description
Mypy is a static type checker for Python. It catches type errors at development time, improving code reliability and enabling better IDE support. Configure it in `pyproject.toml` for consistent type checking.

## Basic Configuration

```toml
# pyproject.toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_ignores = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
strict_equality = true

# Third-party libraries without stubs
[[tool.mypy.overrides]]
module = [
    "boto3.*",
    "redis.*",
]
ignore_missing_imports = true
```

## Strict Mode

```toml
# For new projects or well-typed codebases
[tool.mypy]
strict = true

# Gradual adoption: strict for new modules, relaxed for legacy
[[tool.mypy.overrides]]
module = ["myproject.core.*", "myproject.api.*"]
disallow_untyped_defs = true

[[tool.mypy.overrides]]
module = ["myproject.legacy.*"]
disallow_untyped_defs = false
```

## Type Ignore Comments

```python
# Always use specific error codes
from untyped_lib import process  # type: ignore[import-untyped]

# Document why the ignore is needed
result: list[User] = api.get_users()  # type: ignore[no-any-return]

# Prefer cast() when you know the type
from typing import cast
result = cast(list[User], api.get_users())
```

| Error Code | Meaning |
|------------|---------|
| `import-untyped` | Importing untyped module |
| `no-any-return` | Returning Any from typed function |
| `arg-type` | Argument type mismatch |
| `assignment` | Incompatible assignment |
| `attr-defined` | Attribute not defined |

## Stub Files

```python
# stubs/legacy_lib/__init__.pyi
from typing import Any

class Client:
    def __init__(self, config: dict[str, Any] | None = None) -> None: ...
    def fetch(self, resource: str) -> dict[str, Any]: ...
```

```toml
[tool.mypy]
mypy_path = "stubs"
```

## Commands

```bash
mypy .                    # Type check
mypy . --strict           # Strict mode
mypy --install-types      # Auto-install missing stubs
```

## Notes
- New projects should start with `strict = true`
- Use `mypy.overrides` for per-module settings
- Install type stubs: `pip install types-requests types-redis`
- Enable `warn_unused_ignores` to catch obsolete ignores
- Use `reveal_type(expr)` for debugging type inference

## References
- [Mypy Documentation](https://mypy.readthedocs.io/)
- [Mypy Configuration](https://mypy.readthedocs.io/en/stable/config_file.html)
