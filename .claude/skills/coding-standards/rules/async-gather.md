---
title: Use asyncio.gather for Independent Tasks
impact: HIGH
impactDescription: Significant reduction in I/O wait time
tags: [async, asyncio, concurrency, gather]
---

# Use asyncio.gather for Independent Tasks [HIGH]

## Description
When awaiting multiple independent async operations sequentially, I/O wait times accumulate. Using `asyncio.gather()` runs them concurrently, completing in the time of the slowest operation rather than the sum of all.

## Bad Example
```python
import asyncio

async def fetch_all_data() -> tuple[dict, dict, dict]:
    # Slow: sequential execution accumulates wait time
    users = await fetch_users()       # 1 second
    orders = await fetch_orders()     # 1 second
    products = await fetch_products() # 1 second
    return users, orders, products    # Total: 3 seconds
```

## Good Example
```python
import asyncio

async def fetch_all_data() -> tuple[dict, dict, dict]:
    # Fast: concurrent execution minimizes wait time
    users, orders, products = await asyncio.gather(
        fetch_users(),
        fetch_orders(),
        fetch_products(),
    )
    return users, orders, products  # ~1 second (longest operation only)

# With error handling
async def fetch_all_data_safe() -> tuple[dict | Exception, ...]:
    results = await asyncio.gather(
        fetch_users(),
        fetch_orders(),
        fetch_products(),
        return_exceptions=True,  # Return exceptions as results
    )
    return results
```

## Notes
- `return_exceptions=True` allows retrieving partial results when some tasks fail
- Don't include dependent tasks in `gather` (await prerequisites separately first)
- For many tasks, combine with `asyncio.Semaphore` to limit concurrency
- Python 3.11+ offers `asyncio.TaskGroup` as a safer alternative

## References
- [Python Docs - asyncio.gather](https://docs.python.org/3/library/asyncio-task.html#asyncio.gather)
