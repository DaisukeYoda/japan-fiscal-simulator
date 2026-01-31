---
title: Proper Background Task Creation
impact: HIGH
impactDescription: Prevent GC issues by holding task references
tags: [async, asyncio, task, background]
---

# Proper Background Task Creation [HIGH]

## Description
Tasks created with `asyncio.create_task()` may be garbage collected and cancelled before completion if no reference is held. Additionally, exceptions may be silently swallowed in "fire-and-forget" patterns without proper handling.

## Bad Example
```python
import asyncio

async def main() -> None:
    # Dangerous: no reference to task
    asyncio.create_task(background_work())  # May be GC'd

    # Problem: exceptions are swallowed
    asyncio.create_task(risky_operation())  # Exceptions not reported
```

## Good Example
```python
import asyncio
from collections.abc import Set

# Global task set to hold references
background_tasks: Set[asyncio.Task[None]] = set()

async def run_background_task(coro) -> None:
    task = asyncio.create_task(coro)
    background_tasks.add(task)
    task.add_done_callback(background_tasks.discard)

async def main() -> None:
    # Safe: reference held while task runs
    await run_background_task(background_work())

    # Or explicitly await the result
    task = asyncio.create_task(important_work())
    try:
        result = await task
    except Exception as e:
        logger.error(f"Task failed: {e}")

# Python 3.11+ use TaskGroup
async def main_modern() -> None:
    async with asyncio.TaskGroup() as tg:
        tg.create_task(work_a())
        tg.create_task(work_b())
    # Reaching here = all tasks complete, exceptions propagate automatically
```

## Notes
- `task.add_done_callback()` registers handlers for task completion
- Python 3.11+ `asyncio.TaskGroup` safely manages tasks including exception handling
- Long-running background tasks should be cancellable via `task.cancel()`
- Enable warnings during debug with `asyncio.get_running_loop().set_debug(True)`

## References
- [Python Docs - asyncio.create_task](https://docs.python.org/3/library/asyncio-task.html#asyncio.create_task)
- [PEP 654 - Exception Groups and except*](https://peps.python.org/pep-0654/)
