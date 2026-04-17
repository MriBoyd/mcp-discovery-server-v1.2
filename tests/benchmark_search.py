import time
import statistics
import torch
from src.hybrid_searcher import HybridToolSearcher
from tests.test_hybrid_searcher import TestPerformance

def benchmark_search_only():
    """Benchmark search performance ONLY (excluding indexing)"""
    test = TestPerformance()
    test.setUp()
    
    # Pre-index
    print("Indexing tools...")
    test.searcher.index(test.large_tools)
    
    results = []
    # Warmup
    print("Warmup...")
    test.searcher.search("warmup query")
    
    print("Benchmarking search...")
    for i in range(10):
        start = time.time()
        test.searcher.search("find a tool for data analysis")
        results.append(time.time() - start)
    
    print(f"\nSearch Benchmark Results (100 tools, search only):")
    print(f"  Mean: {statistics.mean(results):.3f}s")
    print(f"  Median: {statistics.median(results):.3f}s")
    print(f"  Min: {min(results):.3f}s")
    print(f"  Max: {max(results):.3f}s")

if __name__ == "__main__":
    benchmark_search_only()
