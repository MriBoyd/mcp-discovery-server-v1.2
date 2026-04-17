# tests/test_performance.py
import time
import statistics
from tests.test_hybrid_searcher import TestPerformance

def benchmark_search_performance():
    """Benchmark search performance with different tool counts"""
    test = TestPerformance()
    test.setUp()
    
    index_times = []
    search_times = []
    exact_times = []
    
    # Warm up
    test.searcher.index(test.large_tools)
    
    for i in range(10):
        # Index time (should be fast if already indexed)
        start_index = time.time()
        test.searcher.index(test.large_tools)
        index_times.append(time.time() - start_index)
        
        # Semantic Search time
        start_search = time.time()
        test.searcher.search("find a tool for data analysis") # Semantic query
        search_times.append(time.time() - start_search)
        
        # Exact Match time
        start_exact = time.time()
        test.searcher.search("tool_5") # Exact name query
        exact_times.append(time.time() - start_exact)
    
    print(f"\nBenchmark Results (100 tools):")
    print(f"Index Mean: {statistics.mean(index_times):.3f}s")
    print(f"Semantic Search Mean: {statistics.mean(search_times):.3f}s")
    print(f"Exact Match Mean: {statistics.mean(exact_times):.3f}s")

if __name__ == "__main__":
    benchmark_search_performance()
