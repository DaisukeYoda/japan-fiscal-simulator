---
title: Use Dataclass for Data Containers
impact: MEDIUM
impactDescription: Less boilerplate, better semantics
tags: [oop, dataclass, data-structures, boilerplate]
---

# Use Dataclass for Data Containers [MEDIUM]

## Description
For classes that primarily hold data, `@dataclass` automatically generates `__init__`, `__repr__`, `__eq__`, and other methods. This reduces boilerplate, prevents bugs, and makes intent clear.

## Bad Example
```python
class User:
    def __init__(self, name: str, email: str, age: int) -> None:
        self.name = name
        self.email = email
        self.age = age

    def __repr__(self) -> str:
        return f"User(name={self.name!r}, email={self.email!r}, age={self.age})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, User):
            return NotImplemented
        return self.name == other.name and self.email == other.email and self.age == other.age

    def __hash__(self) -> int:
        return hash((self.name, self.email, self.age))
```

## Good Example
```python
from dataclasses import dataclass, field

@dataclass
class User:
    name: str
    email: str
    age: int

# With defaults and computed fields
@dataclass
class Order:
    items: list[Item]
    customer_id: str
    discount: float = 0.0
    _total: float = field(init=False)

    def __post_init__(self) -> None:
        self._total = sum(item.price for item in self.items) * (1 - self.discount)

# Immutable dataclass
@dataclass(frozen=True)
class Point:
    x: float
    y: float

# With slots for memory efficiency (Python 3.10+)
@dataclass(slots=True)
class Coordinate:
    lat: float
    lon: float
```

## Notes
- Use `frozen=True` for immutable objects that can be hashed and used as dict keys
- Use `slots=True` (Python 3.10+) for memory efficiency with many instances
- Use `field(default_factory=list)` for mutable default values
- Consider `attrs` library for more features or `pydantic` for validation

## References
- [Python Docs - dataclasses](https://docs.python.org/3/library/dataclasses.html)
- [PEP 557 - Data Classes](https://peps.python.org/pep-0557/)
