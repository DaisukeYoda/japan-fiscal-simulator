---
title: Single Responsibility Principle
impact: HIGH
impactDescription: Improved maintainability and testability
tags: [design, solid, srp, maintainability]
---

# Single Responsibility Principle [HIGH]

## Description
A class or function should have only one reason to change. When a unit of code handles multiple responsibilities, changes to one responsibility may inadvertently affect others, making the code harder to maintain and test.

## Bad Example
```python
class UserManager:
    def __init__(self, db_connection) -> None:
        self.db = db_connection

    def create_user(self, email: str, password: str) -> User:
        # Responsibility 1: Validation
        if not self._validate_email(email):
            raise ValueError("Invalid email")

        # Responsibility 2: Password hashing
        hashed = hashlib.sha256(password.encode()).hexdigest()

        # Responsibility 3: Database operations
        self.db.execute("INSERT INTO users ...", (email, hashed))

        # Responsibility 4: Email notification
        self._send_welcome_email(email)

        return User(email=email)

    def _validate_email(self, email: str) -> bool: ...
    def _send_welcome_email(self, email: str) -> None: ...
```

## Good Example
```python
from dataclasses import dataclass

@dataclass
class UserValidator:
    def validate_email(self, email: str) -> bool:
        return "@" in email and "." in email.split("@")[1]

@dataclass
class PasswordHasher:
    def hash(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

@dataclass
class UserRepository:
    db: Connection

    def save(self, user: User) -> None:
        self.db.execute("INSERT INTO users ...", (user.email, user.password_hash))

@dataclass
class EmailService:
    def send_welcome(self, email: str) -> None: ...

class UserService:
    def __init__(
        self,
        validator: UserValidator,
        hasher: PasswordHasher,
        repository: UserRepository,
        email_service: EmailService,
    ) -> None:
        self.validator = validator
        self.hasher = hasher
        self.repository = repository
        self.email_service = email_service

    def create_user(self, email: str, password: str) -> User:
        if not self.validator.validate_email(email):
            raise ValueError("Invalid email")

        user = User(email=email, password_hash=self.hasher.hash(password))
        self.repository.save(user)
        self.email_service.send_welcome(email)
        return user
```

## Notes
- Signs of SRP violation: "and" in class descriptions, multiple unrelated imports, large classes
- Each component can now be tested in isolation
- Changes to email logic won't affect password hashing
- Enables easier dependency injection and mocking

## References
- [Clean Code by Robert C. Martin](https://www.oreilly.com/library/view/clean-code-a/9780136083238/)
