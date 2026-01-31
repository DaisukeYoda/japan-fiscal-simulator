---
title: Testing with Pytest
impact: HIGH
impactDescription: Reliable test execution
tags: [pytest, testing, fixtures, coverage]
---

# Testing with Pytest [HIGH]

## Description
Pytest is the standard Python testing framework. Configure it in `pyproject.toml` for consistent test execution, effective fixtures, parametrized tests, and coverage reporting.

## Basic Configuration

```toml
# pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "-ra",
]
markers = [
    "slow: marks tests as slow",
    "integration: marks integration tests",
]
filterwarnings = [
    "error",
    "ignore::DeprecationWarning",
]
```

## Fixtures

```python
# conftest.py
import pytest
from typing import Generator

@pytest.fixture
def db() -> Generator[Database, None, None]:
    """Database connection with automatic cleanup."""
    database = Database("test://localhost")
    database.connect()
    yield database
    database.close()

@pytest.fixture
def user(db: Database) -> User:
    """Test user saved to database."""
    user = User(name="Alice", email="alice@example.com")
    db.save(user)
    return user

# Factory fixture for multiple instances
@pytest.fixture
def create_user(db: Database):
    """Factory for creating test users."""
    created: list[User] = []

    def _create_user(name: str, email: str) -> User:
        user = User(name=name, email=email)
        db.save(user)
        created.append(user)
        return user

    yield _create_user

    for user in created:
        db.delete(user)
```

| Scope | Lifetime |
|-------|----------|
| `function` | Each test (default) |
| `class` | All tests in class |
| `module` | All tests in file |
| `session` | Entire test run |

## Parametrize

```python
import pytest

@pytest.mark.parametrize(
    ("email", "expected"),
    [
        ("user@example.com", True),
        ("user@sub.example.com", True),
        ("invalid", False),
        ("", False),
    ],
    ids=["valid", "subdomain", "no_at", "empty"],
)
def test_validate_email(email: str, expected: bool) -> None:
    assert validate_email(email) is expected

# Multiple parameters (combinations)
@pytest.mark.parametrize("x", [1, 2])
@pytest.mark.parametrize("y", [10, 20])
def test_multiply(x: int, y: int) -> None:
    assert multiply(x, y) == x * y
```

## Coverage

```toml
[tool.pytest.ini_options]
addopts = ["--cov=src", "--cov-report=term-missing"]

[tool.coverage.run]
source = ["src"]
branch = true
omit = ["*/tests/*", "*/__main__.py"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "@abstractmethod",
]
fail_under = 80
```

## Commands

```bash
pytest                      # Run all tests
pytest -x                   # Stop on first failure
pytest -k "test_user"       # Run matching tests
pytest -m "not slow"        # Skip slow tests
pytest --cov=src            # With coverage
pytest --cov-report=html    # HTML coverage report
```

## Notes
- Put shared fixtures in `conftest.py` (auto-discovered)
- Use `yield` for setup/teardown in one fixture
- Use `ids` parameter for readable test names
- `branch = true` catches untested if/else branches
- 80% coverage is reasonable; don't chase 100%

## References
- [Pytest Documentation](https://docs.pytest.org/)
- [pytest-cov](https://pytest-cov.readthedocs.io/)
