import unittest
import asyncio
import time
from src.circuit_breaker import CircuitBreaker, State, CircuitOpenError, CircuitBreakerManager

class TestCircuitBreaker(unittest.IsolatedAsyncioTestCase):
    async def test_initial_state(self):
        cb = CircuitBreaker()
        self.assertEqual(cb.state, State.CLOSED)

    async def test_success_keeps_closed(self):
        cb = CircuitBreaker()
        async def success_func(): return "success"
        
        result = await cb.call(success_func)
        self.assertEqual(result, "success")
        self.assertEqual(cb.state, State.CLOSED)

    async def test_failure_threshold_trips_open(self):
        cb = CircuitBreaker(failure_threshold=2)
        async def fail_func(): raise ValueError("fail")
        
        with self.assertRaises(ValueError):
            await cb.call(fail_func)
        self.assertEqual(cb.state, State.CLOSED)
        
        with self.assertRaises(ValueError):
            await cb.call(fail_func)
        self.assertEqual(cb.state, State.OPEN)

    async def test_open_prevents_calls(self):
        cb = CircuitBreaker(failure_threshold=1)
        async def fail_func(): raise ValueError("fail")
        
        with self.assertRaises(ValueError):
            await cb.call(fail_func)
        
        with self.assertRaises(CircuitOpenError):
            await cb.call(fail_func)

    async def test_recovery_to_half_open(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        async def fail_func(): raise ValueError("fail")
        
        with self.assertRaises(ValueError):
            await cb.call(fail_func)
        
        self.assertEqual(cb.state, State.OPEN)
        await asyncio.sleep(0.15)
        
        async def success_func(): return "success"
        result = await cb.call(success_func)
        self.assertEqual(result, "success")
        self.assertEqual(cb.state, State.HALF_OPEN)

    async def test_half_open_to_closed(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1, half_open_success=2)
        async def fail_func(): raise ValueError("fail")
        await self.assertRaises(ValueError, cb.call, fail_func)
        
        await asyncio.sleep(0.15)
        
        async def success_func(): return "success"
        await cb.call(success_func) # 1st success -> HALF_OPEN
        self.assertEqual(cb.state, State.HALF_OPEN)
        
        await cb.call(success_func) # 2nd success -> CLOSED
        self.assertEqual(cb.state, State.CLOSED)

    async def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)
        async def fail_func(): raise ValueError("fail")
        await self.assertRaises(ValueError, cb.call, fail_func)
        
        await asyncio.sleep(0.15)
        
        with self.assertRaises(ValueError):
            await cb.call(fail_func)
        self.assertEqual(cb.state, State.OPEN)

class TestCircuitBreakerManager(unittest.IsolatedAsyncioTestCase):
    async def test_manager_gets_named_breaker(self):
        manager = CircuitBreakerManager(failure_threshold=3)
        cb1 = await manager.get_breaker("server1")
        cb2 = await manager.get_breaker("server2")
        
        self.assertNotEqual(cb1, cb2)
        self.assertEqual(cb1.failure_threshold, 3)
        self.assertEqual(cb2.failure_threshold, 3)

    async def test_manager_call(self):
        manager = CircuitBreakerManager()
        async def test_func(x): return x * 2
        
        result = await manager.call("server1", test_func, 5)
        self.assertEqual(result, 10)

if __name__ == "__main__":
    unittest.main()
