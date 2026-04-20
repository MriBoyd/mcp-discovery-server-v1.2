import unittest
import asyncio
import time
from src.rate_limiter import TokenBucketRateLimiter, RateLimiterManager

class TestRateLimiter(unittest.IsolatedAsyncioTestCase):
    async def test_try_acquire_success(self):
        # 10 tokens per sec, capacity 5
        limiter = TokenBucketRateLimiter(rate=10.0, capacity=5)
        
        # Initial capacity is full
        self.assertTrue(await limiter.try_acquire(3))
        self.assertTrue(await limiter.try_acquire(2))
        # Now empty
        self.assertFalse(await limiter.try_acquire(1))

    async def test_refill(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=10)
        # Empty it
        await limiter.try_acquire(10)
        self.assertFalse(await limiter.try_acquire(1))
        
        # Wait for refill (0.02s should give 2 tokens)
        await asyncio.sleep(0.02)
        self.assertTrue(await limiter.try_acquire(1))

    async def test_acquire_wait(self):
        limiter = TokenBucketRateLimiter(rate=100.0, capacity=1)
        await limiter.try_acquire(1) # Empty it
        
        start = time.monotonic()
        await limiter.acquire(1) # Should wait ~0.01s
        elapsed = time.monotonic() - start
        
        self.assertGreaterEqual(elapsed, 0.005)

class TestRateLimiterManager(unittest.IsolatedAsyncioTestCase):
    async def test_manager_limiters(self):
        manager = RateLimiterManager(rate=10.0, capacity=5)
        l1 = await manager.get_limiter("user1")
        l2 = await manager.get_limiter("user2")
        
        self.assertNotEqual(l1, l2)
        self.assertEqual(l1.rate, 10.0)

    async def test_manager_try_acquire(self):
        manager = RateLimiterManager(rate=1.0, capacity=1)
        self.assertTrue(await manager.try_acquire("user1"))
        self.assertFalse(await manager.try_acquire("user1"))
        self.assertTrue(await manager.try_acquire("user2"))

if __name__ == "__main__":
    unittest.main()
