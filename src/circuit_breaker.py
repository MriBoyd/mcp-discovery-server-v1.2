# circuit_breaker.py

import asyncio
import time
from enum import Enum


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_success: int = 2,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_success = half_open_success

        self._state = State.CLOSED
        self._failures = 0
        self._last_failure_time = 0
        self._success_count = 0

        self._lock = asyncio.Lock()

    async def call(self, func, *args, **kwargs):
        async with self._lock:
            now = time.time()

            # OPEN → check if we can move to HALF_OPEN
            if self._state == State.OPEN:
                if now - self._last_failure_time > self.recovery_timeout:
                    self._state = State.HALF_OPEN
                    self._success_count = 0
                else:
                    raise Exception("Circuit is OPEN")

        # execute outside lock
        try:
            result = await func(*args, **kwargs)
        except Exception:
            await self._on_failure()
            raise
        else:
            await self._on_success()
            return result

    async def _on_failure(self):
        async with self._lock:
            self._failures += 1
            self._last_failure_time = time.time()

            if self._failures >= self.failure_threshold:
                self._state = State.OPEN

    async def _on_success(self):
        async with self._lock:
            if self._state == State.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_success:
                    self._reset()
            else:
                self._reset()

    def _reset(self):
        self._state = State.CLOSED
        self._failures = 0
        self._success_count = 0

    def state(self):
        return self._state