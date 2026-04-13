# tests/test_performance.py
import time
import statistics
from tests.test_hybrid_searcher import TestPerformance

def benchmark_search_performance():
    """Benchmark search performance with different tool counts"""
    test = TestPerformance()
    test.setUp()
    
    results = []
    for i in range(10):
        start = time.time()
        test.searcher.index(test.large_tools)
        search_start = time.time()
        test.searcher.search("test query")
        total_time = time.time() - start
        results.append(total_time)
    
    print(f"\nBenchmark Results (100 tools):")
    print(f"  Mean: {statistics.mean(results):.3f}s")
    print(f"  Median: {statistics.median(results):.3f}s")
    print(f"  Min: {min(results):.3f}s")
    print(f"  Max: {max(results):.3f}s")

if __name__ == "__main__":
    benchmark_search_performance()