# rate_limiter.py

import asyncio
import time


class TokenBucketRateLimiter:
    def __init__(self, rate: float, capacity: int):
        """
        rate: tokens per second
        capacity: max burst size
        """
        self.rate = rate
        self.capacity = capacity

        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self._last_refill

                # refill tokens
                self._tokens = min(
                    self.capacity,
                    self._tokens + elapsed * self.rate
                )
                self._last_refill = now

                if self._tokens >= 1:
                    self._tokens -= 1
                    return

                # wait for next token
                wait_time = (1 - self._tokens) / self.rate
                await asyncio.sleep(wait_time)