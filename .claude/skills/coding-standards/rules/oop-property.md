---
title: Use Property Instead of Getters
impact: MEDIUM
impactDescription: Pythonic attribute access with encapsulation
tags: [oop, property, encapsulation, pythonic]
---

# Use Property Instead of Getters [MEDIUM]

## Description
Python's `@property` decorator provides attribute-style access while allowing custom logic. This is more Pythonic than Java-style getter/setter methods and allows adding validation or computation without changing the API.

## Bad Example
```python
class Circle:
    def __init__(self, radius: float) -> None:
        self._radius = radius

    # Java-style: verbose and unpythonic
    def get_radius(self) -> float:
        return self._radius

    def set_radius(self, value: float) -> None:
        if value < 0:
            raise ValueError("Radius must be non-negative")
        self._radius = value

    def get_area(self) -> float:
        return 3.14159 * self._radius ** 2

# Usage is clunky
circle = Circle(5)
r = circle.get_radius()
circle.set_radius(10)
area = circle.get_area()
```

## Good Example
```python
import math

class Circle:
    def __init__(self, radius: float) -> None:
        self._radius = radius

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        if value < 0:
            raise ValueError("Radius must be non-negative")
        self._radius = value

    @property
    def area(self) -> float:
        """Computed property: calculated on access."""
        return math.pi * self._radius ** 2

    @property
    def diameter(self) -> float:
        return self._radius * 2

# Clean attribute-style access
circle = Circle(5)
r = circle.radius
circle.radius = 10
area = circle.area  # Computed transparently

# Read-only property (no setter)
class User:
    def __init__(self, first: str, last: str) -> None:
        self.first = first
        self.last = last

    @property
    def full_name(self) -> str:
        return f"{self.first} {self.last}"

# Using cached_property for expensive computations
from functools import cached_property

class DataAnalysis:
    def __init__(self, data: list[float]) -> None:
        self.data = data

    @cached_property
    def statistics(self) -> dict[str, float]:
        """Computed once and cached."""
        return {
            "mean": sum(self.data) / len(self.data),
            "max": max(self.data),
            "min": min(self.data),
        }
```

## Notes
- Properties start simple and add validation later without API changes
- Use `@cached_property` (Python 3.8+) for expensive computations accessed multiple times
- Read-only properties (no setter) prevent accidental modification
- Avoid heavy computation in properties unless cached—users expect fast attribute access

## References
- [Python Docs - property](https://docs.python.org/3/library/functions.html#property)
- [Python Docs - functools.cached_property](https://docs.python.org/3/library/functools.html#functools.cached_property)
