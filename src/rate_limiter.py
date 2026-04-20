# rate_limiter.py

import asyncio
import time
from typing import Dict


class TokenBucketRateLimiter:
    def __init__(self, rate: float, capacity: int):
        """
        rate: tokens per second
        capacity: max burst size
        """
        self.rate = rate
        self.capacity = capacity

        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def _refill(self):
        now = time.monotonic()
        elapsed = now - self._last_refill
        
        # Add new tokens based on elapsed time
        self._tokens = min(
            self.capacity,
            self._tokens + elapsed * self.rate
        )
        self._last_refill = now

    async def try_acquire(self, tokens: int = 1) -> bool:
        """Attempt to acquire tokens without waiting. Returns True if successful."""
        async with self._lock:
            await self._refill()
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    async def acquire(self, tokens: int = 1):
        """Acquire tokens, waiting if necessary."""
        async with self._lock:
            while True:
                await self._refill()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                # Wait for next token
                wait_time = (tokens - self._tokens) / self.rate
                await asyncio.sleep(wait_time)


class RateLimiterManager:
    """Manages rate limiters for multiple entities (e.g., users, servers)"""
    
    def __init__(self, rate: float, capacity: int):
        self.rate = rate
        self.capacity = capacity
        self.limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._lock = asyncio.Lock()

    async def get_limiter(self, key: str) -> TokenBucketRateLimiter:
        async with self._lock:
            if key not in self.limiters:
                self.limiters[key] = TokenBucketRateLimiter(self.rate, self.capacity)
            return self.limiters[key]

    async def try_acquire(self, key: str, tokens: int = 1) -> bool:
        limiter = await self.get_limiter(key)
        return await limiter.try_acquire(tokens)

    async def acquire(self, key: str, tokens: int = 1):
        limiter = await self.get_limiter(key)
        await limiter.acquire(tokens)
