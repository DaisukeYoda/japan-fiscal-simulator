---
title: Limit Concurrency with Semaphores
impact: HIGH
impactDescription: Prevent resource exhaustion and rate limiting issues
tags: [async, asyncio, semaphore, concurrency-control]
---

# Limit Concurrency with Semaphores [HIGH]

## Description
When processing many async operations concurrently, unbounded concurrency can exhaust resources (connections, file descriptors) or trigger rate limits. `asyncio.Semaphore` limits the number of concurrent operations.

## Bad Example
```python
import asyncio

async def fetch_all_urls(urls: list[str]) -> list[dict]:
    # Dangerous: 10,000 concurrent connections
    tasks = [fetch_url(url) for url in urls]  # len(urls) = 10,000
    return await asyncio.gather(*tasks)  # May exhaust connections or hit rate limits
```

## Good Example
```python
import asyncio

async def fetch_all_urls(urls: list[str], max_concurrent: int = 10) -> list[dict]:
    semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch_with_limit(url: str) -> dict:
        async with semaphore:  # Only max_concurrent tasks run at once
            return await fetch_url(url)

    tasks = [fetch_with_limit(url) for url in urls]
    return await asyncio.gather(*tasks)

# Reusable pattern with class
class RateLimitedClient:
    def __init__(self, max_concurrent: int = 10) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def fetch(self, url: str) -> dict:
        async with self._semaphore:
            return await self._do_fetch(url)

# Using asyncio.BoundedSemaphore for stricter control
bounded_sem = asyncio.BoundedSemaphore(10)  # Raises if released more than acquired
```

## Notes
- `Semaphore` allows releasing more times than acquired; use `BoundedSemaphore` to prevent this
- Choose the concurrency limit based on the resource constraints (API rate limits, connection pools, etc.)
- Consider using `asyncio.Queue` for producer-consumer patterns with more complex flow control
- For HTTP clients, many libraries have built-in connection limits (e.g., `aiohttp.TCPConnector(limit=10)`)

## References
- [Python Docs - asyncio.Semaphore](https://docs.python.org/3/library/asyncio-sync.html#asyncio.Semaphore)
