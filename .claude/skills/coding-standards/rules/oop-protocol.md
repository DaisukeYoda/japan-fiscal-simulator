---
title: Prefer Protocol Over Abstract Base Classes
impact: MEDIUM
impactDescription: Structural typing without inheritance requirements
tags: [oop, protocol, typing, duck-typing]
---

# Prefer Protocol Over Abstract Base Classes [MEDIUM]

## Description
`Protocol` enables structural subtyping (duck typing with static type checking). Unlike abstract base classes (ABC), classes don't need to explicitly inherit from a Protocol—they just need to implement the required methods. This provides flexibility while maintaining type safety.

## Bad Example
```python
from abc import ABC, abstractmethod

# Forces explicit inheritance
class Serializable(ABC):
    @abstractmethod
    def to_json(self) -> str: ...

# Every class must explicitly inherit
class User(Serializable):  # Must inherit
    def __init__(self, name: str) -> None:
        self.name = name

    def to_json(self) -> str:
        return f'{{"name": "{self.name}"}}'

# Can't use third-party classes that happen to have to_json()
def save(obj: Serializable) -> None:  # Only accepts Serializable subclasses
    data = obj.to_json()
    # ...
```

## Good Example
```python
from typing import Protocol

# No inheritance required
class Serializable(Protocol):
    def to_json(self) -> str: ...

# Works without inheriting from Serializable
class User:
    def __init__(self, name: str) -> None:
        self.name = name

    def to_json(self) -> str:
        return f'{{"name": "{self.name}"}}'

# Any object with to_json() works
def save(obj: Serializable) -> None:
    data = obj.to_json()
    # ...

# Third-party classes work if they have the method
save(User("Alice"))  # Works
save(some_third_party_object_with_to_json)  # Also works!

# Protocols can have multiple methods
class Repository(Protocol):
    def get(self, id: str) -> Entity: ...
    def save(self, entity: Entity) -> None: ...
    def delete(self, id: str) -> None: ...

# Runtime checking with @runtime_checkable
from typing import runtime_checkable

@runtime_checkable
class Closable(Protocol):
    def close(self) -> None: ...

if isinstance(resource, Closable):
    resource.close()
```

## Notes
- Protocols work at type-check time; use `@runtime_checkable` for `isinstance()` checks
- Prefer Protocol for interfaces that external code might implement
- Use ABC when you need shared implementation or want to enforce inheritance
- Protocol supports properties, class methods, and attributes

## References
- [Python Docs - typing.Protocol](https://docs.python.org/3/library/typing.html#typing.Protocol)
- [PEP 544 - Protocols: Structural subtyping](https://peps.python.org/pep-0544/)
