# circuit_breaker.py

import asyncio
import time
from enum import Enum
from typing import Dict, Callable, Any


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(Exception):
    """Raised when the circuit is open and refusing requests."""
    pass


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

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        async with self._lock:
            now = time.time()

            # OPEN → check if we can move to HALF_OPEN
            if self._state == State.OPEN:
                if now - self._last_failure_time > self.recovery_timeout:
                    self._state = State.HALF_OPEN
                    self._success_count = 0
                else:
                    raise CircuitOpenError(f"Circuit is OPEN. Remaining recovery time: {int(self.recovery_timeout - (now - self._last_failure_time))}s")

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
            if self._state == State.HALF_OPEN:
                # If we fail in half-open, immediately go back to OPEN
                self._state = State.OPEN
            else:
                self._failures += 1
                if self._failures >= self.failure_threshold:
                    self._state = State.OPEN
            
            self._last_failure_time = time.time()

    async def _on_success(self):
        async with self._lock:
            if self._state == State.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_success:
                    self._reset()
            elif self._state == State.CLOSED:
                # Normal success, just ensure failures are reset if they hadn't tripped yet
                # (Optional: some implementations only reset on threshold trip, but resetting on any success is safer)
                self._failures = 0

    def _reset(self):
        self._state = State.CLOSED
        self._failures = 0
        self._success_count = 0

    @property
    def state(self):
        return self._state


class CircuitBreakerManager:
    """Manages circuit breakers for multiple named resources (e.g., servers)"""
    
    def __init__(self, **default_kwargs):
        self.default_kwargs = default_kwargs
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get_breaker(self, name: str) -> CircuitBreaker:
        async with self._lock:
            if name not in self.breakers:
                self.breakers[name] = CircuitBreaker(**self.default_kwargs)
            return self.breakers[name]

    async def call(self, name: str, func: Callable, *args, **kwargs) -> Any:
        breaker = await self.get_breaker(name)
        return await breaker.call(func, *args, **kwargs)
