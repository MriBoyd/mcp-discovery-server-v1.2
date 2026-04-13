# tests/test_hybrid_searcher.py
import unittest
import json
import tempfile
import os
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hybrid_searcher import HybridToolSearcher
from src.config import Config

class TestHybridToolSearcher(unittest.TestCase):
    """Unit tests for HybridToolSearcher"""
    
    def setUp(self):
        """Set up test fixtures before each test"""
        self.sample_tools = [
            {
                "name": "create_github_issue",
                "description": "Create a new issue in a GitHub repository",
                "parameters": {
                    "repo": {"type": "string", "description": "Repository name"},
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue description"}
                },
                "required": ["repo", "title"]
            },
            {
                "name": "send_slack_message",
                "description": "Send a message to a Slack channel",
                "parameters": {
                    "channel": {"type": "string", "description": "Channel ID"},
                    "text": {"type": "string", "description": "Message text"}
                },
                "required": ["channel", "text"]
            },
            {
                "name": "execute_sql_query",
                "description": "Execute a SQL query on the database",
                "parameters": {
                    "query": {"type": "string", "description": "SQL query"},
                    "database": {"type": "string", "description": "Database name"}
                },
                "required": ["query"]
            }
        ]
        
        self.searcher = HybridToolSearcher()
    
    def test_initialization(self):
        """Test that searcher initializes correctly"""
        self.assertIsNotNone(self.searcher.embedder)
        self.assertIsNotNone(self.searcher.reranker)
        self.assertIsNone(self.searcher.bm25)
        self.assertIsNone(self.searcher.tool_embeddings)
        self.assertFalse(self.searcher.is_indexed)
    
    def test_prepare_tool_text(self):
        """Test tool text preparation"""
        tool = self.sample_tools[0]
        text = self.searcher._prepare_tool_text(tool)
        
        # Check required content
        self.assertIn("Function: create_github_issue", text)
        self.assertIn("Description: Create a new issue", text)
        self.assertIn("Parameters:", text)
        self.assertIn("repo (string): Repository name", text)
        self.assertIn("title (string): Issue title", text)
        self.assertIn("Required parameters: repo, title", text)
    
    def test_index_tools(self):
        """Test indexing tools"""
        self.searcher.index(self.sample_tools)
        if self.searcher.tool_embeddings is not None:
            print(self.searcher.tool_embeddings.shape)
        # Check indexing results
        self.assertTrue(self.searcher.is_indexed)
        self.assertIsNotNone(self.searcher.bm25)
        self.assertIsNotNone(self.searcher.tool_embeddings)
        self.assertEqual(len(self.searcher.tools), 3)
        self.assertEqual(len(self.searcher.tool_texts), 3)
        
        # Check embeddings shape
        self.assertIsNotNone(self.searcher.tool_embeddings)
        self.assertIsInstance(self.searcher.tool_embeddings, np.ndarray)
        self.assertEqual(self.searcher.tool_embeddings.shape[0], 3)
        self.assertEqual(self.searcher.tool_embeddings.shape[1], Config.EMBEDDING_DIM)
    
    def test_bm25_search(self):
        """Test BM25 lexical search"""
        self.searcher.index(self.sample_tools)
        
        # Test exact match
        results = self.searcher._bm25_search("create github issue", top_k=2)
        self.assertGreater(len(results), 0)
        
        # Check result format
        for score, idx in results:
            self.assertIsInstance(score, float)
            self.assertIsInstance(idx, int)
            self.assertGreaterEqual(score, 0)
    
    def test_dense_search(self):
        """Test dense semantic search"""
        self.searcher.index(self.sample_tools)
        
        # Test semantic search
        results = self.searcher._dense_search("create a bug report", top_k=2)
        self.assertGreater(len(results), 0)
        
        # Check result format
        for score, idx in results:
            self.assertIsInstance(score, float)
            self.assertIsInstance(idx, int)
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1.0)  # Cosine similarity range
    
    def test_fuse_scores(self):
        """Test score fusion"""
        self.searcher.index(self.sample_tools)
        
        bm25_results = [(0.8, 0), (0.6, 1)]
        dense_results = [(0.7, 0), (0.5, 2)]
        
        fused = self.searcher._fuse_scores(bm25_results, dense_results)
        
        # Check fusion
        self.assertGreater(len(fused), 0)
        for score, idx in fused:
            self.assertIsInstance(score, float)
            self.assertIsInstance(idx, int)
    
    def test_search(self):
        """Test complete search pipeline"""
        self.searcher.index(self.sample_tools)
        
        # Test search with various queries
        test_queries = [
            "create a github issue",
            "send message to slack",
            "run sql query"
        ]
        
        for query in test_queries:
            results = self.searcher.search(query, top_k=2)
            
            # Check results
            self.assertIsInstance(results, list)
            if len(results) > 0:
                result = results[0]
                self.assertIn('tool_name', result)
                self.assertIn('tool_description', result)
                self.assertIn('tool_schema', result)
                self.assertIn('relevance_score', result)
                self.assertIsInstance(result['relevance_score'], float)
    
    def test_search_without_indexing(self):
        """Test that search fails if index not called"""
        with self.assertRaises(RuntimeError):
            self.searcher.search("test query")
    
    def test_search_empty_results(self):
        """Test search with no matching tools"""
        # Create searcher with empty tool list
        empty_searcher = HybridToolSearcher()
        empty_searcher.index([])
        
        results = empty_searcher.search("test query")
        self.assertEqual(len(results), 0)
    
    def test_bm25_no_results(self):
        """Test BM25 with no matching terms"""
        self.searcher.index(self.sample_tools)
        
        # Query with completely different terms
        results = self.searcher._bm25_search("xyzabc123", top_k=2)
        
        # Should return empty or very low scores
        for score, idx in results:
            self.assertLessEqual(score, 0.5)
    
    def test_tool_text_formatting(self):
        """Test that tool text includes all required fields"""
        tool = {
            "name": "test_tool",
            "description": "Test description",
            "parameters": {
                "param1": {"type": "string", "description": "Test param"}
            },
            "required": ["param1"]
        }
        
        text = self.searcher._prepare_tool_text(tool)
        
        self.assertIn("test_tool", text)
        self.assertIn("Test description", text)
        self.assertIn("param1", text)
        self.assertIn("string", text)
        self.assertIn("required", text.lower())
    
    @patch('src.hybrid_searcher.CodeEmbedder')
    def test_embedder_initialization(self, mock_embedder):
        """Test embedder initialization"""
        mock_embedder.return_value = Mock()
        searcher = HybridToolSearcher()
        self.assertIsNotNone(searcher.embedder)
    
    def test_numpy_array_conversion(self):
        """Test that embeddings are stored as numpy arrays"""
        self.searcher.index(self.sample_tools)
        
        self.assertIsInstance(self.searcher.tool_embeddings, np.ndarray)
        self.assertEqual(self.searcher.tool_embeddings.dtype, np.float32)


class TestConfig(unittest.TestCase):
    """Test configuration settings"""
    
    def test_config_values(self):
        """Test config has required attributes"""
        self.assertTrue(hasattr(Config, 'MODEL_CACHE_DIR'))
        self.assertTrue(hasattr(Config, 'CODE_EMBEDDING_MODEL'))
        self.assertTrue(hasattr(Config, 'EMBEDDING_DIM'))
        self.assertTrue(hasattr(Config, 'BM25_CANDIDATES'))
        self.assertTrue(hasattr(Config, 'DENSE_CANDIDATES'))
        self.assertTrue(hasattr(Config, 'FUSION_CANDIDATES'))
        self.assertTrue(hasattr(Config, 'FINAL_RESULTS'))
        self.assertTrue(hasattr(Config, 'BM25_WEIGHT'))
        self.assertTrue(hasattr(Config, 'DENSE_WEIGHT'))
    
    def test_embedding_dimension(self):
        """Test embedding dimension is correct for Jina model"""
        self.assertEqual(Config.EMBEDDING_DIM, 1536)
    
    def test_fusion_weights(self):
        """Test fusion weights sum to 1.0"""
        total = Config.BM25_WEIGHT + Config.DENSE_WEIGHT
        self.assertAlmostEqual(total, 1.0, places=5)


class TestPerformance(unittest.TestCase):
    """Performance tests for large-scale operations"""
    
    def setUp(self):
        self.searcher = HybridToolSearcher()
        
        # Create larger test dataset
        self.large_tools = []
        for i in range(100):  # 100 tools for performance test
            tool = {
                "name": f"tool_{i}",
                "description": f"Description for tool {i} " + " ".join([f"keyword{j}" for j in range(5)]),
                "parameters": {
                    f"param_{j}": {"type": "string", "description": f"Parameter {j}"}
                    for j in range(3)
                },
                "required": ["param_0"]
            }
            self.large_tools.append(tool)
    
    def test_indexing_performance(self):
        """Test indexing time for 100 tools"""
        import time
        
        start = time.time()
        self.searcher.index(self.large_tools)
        duration = time.time() - start
        
        print(f"Indexing 100 tools took {duration:.2f} seconds")
        self.assertLess(duration, 60)  # Should take less than 60 seconds
    
    def test_search_performance(self):
        """Test search time after indexing"""
        self.searcher.index(self.large_tools)
        
        import time
        start = time.time()
        results = self.searcher.search("test query", top_k=5)
        duration = time.time() - start
        
        print(f"Search took {duration:.3f} seconds")
        self.assertLess(duration, 5)  # Should take less than 5 seconds


class TestIntegration(unittest.TestCase):
    """Integration tests with real queries"""
    
    def setUp(self):
        self.searcher = HybridToolSearcher()
        
        # Load real tools if available, otherwise use samples
        self.tools = []
        tools_path = "tools/all_tools.json"
        if os.path.exists(tools_path):
            with open(tools_path, 'r') as f:
                data = json.load(f)
                self.tools = data if isinstance(data, list) else data.get('tools', [])
        
        if not self.tools:
            # Use sample tools
            self.tools = [
                {
                    "name": "create_github_issue",
                    "description": "Create a new issue in a GitHub repository",
                    "parameters": {
                        "repo": {"type": "string", "description": "Repository name"},
                        "title": {"type": "string", "description": "Issue title"}
                    },
                    "required": ["repo", "title"]
                },
                {
                    "name": "search_github_code",
                    "description": "Search for code in GitHub repositories",
                    "parameters": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            ]
        
        self.searcher.index(self.tools)
    
    def test_semantic_search_github(self):
        """Test semantic understanding of GitHub-related queries"""
        results = self.searcher.search("I need to report a bug in my repository", top_k=3)
        
        # Should find GitHub-related tools
        found_github = any('github' in r['tool_name'].lower() for r in results)
        self.assertTrue(found_github, "Should find GitHub tools for bug report query")
    
    def test_exact_match_priority(self):
        """Test that exact matches are prioritized"""
        results = self.searcher.search("create_github_issue", top_k=3)
        
        if results:
            # Exact match should be first
            self.assertEqual(results[0]['tool_name'], "create_github_issue")
    
    def test_query_diversity(self):
        """Test different query types"""
        queries = [
            "search code",
            "create issue",
            "find repository",
            "bug tracker"
        ]
        
        for query in queries:
            results = self.searcher.search(query, top_k=2)
            self.assertIsInstance(results, list)
            # Should return at least one result for common queries
            if len(self.tools) > 5:
                self.assertGreater(len(results), 0, f"No results for query: {query}")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""
    
    def setUp(self):
        self.sample_tools = [
            {
                "name": "create_github_issue",
                "description": "Create a new issue in a GitHub repository",
                "parameters": {
                    "repo": {"type": "string", "description": "Repository name"},
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue description"}
                },
                "required": ["repo", "title"]
            },
            {
                "name": "send_slack_message",
                "description": "Send a message to a Slack channel",
                "parameters": {
                    "channel": {"type": "string", "description": "Channel ID"},
                    "text": {"type": "string", "description": "Message text"}
                },
                "required": ["channel", "text"]
            },
            {
                "name": "execute_sql_query",
                "description": "Execute a SQL query on the database",
                "parameters": {
                    "query": {"type": "string", "description": "SQL query"},
                    "database": {"type": "string", "description": "Database name"}
                },
                "required": ["query"]
            }
        ]
        self.searcher = HybridToolSearcher()
    
    def test_empty_tools_list(self):
        """Test indexing empty tool list"""
        self.searcher.index([])
        self.assertTrue(self.searcher.is_indexed)
        results = self.searcher.search("test")
        self.assertEqual(len(results), 0)
    
    def test_empty_query(self):
        """Test search with empty query"""
        self.searcher.index([{"name": "test", "description": "test", "parameters": {}}])
        results = self.searcher.search("")
        self.assertIsInstance(results, list)
    
    def test_special_characters_in_query(self):
        """Test search with special characters"""
        self.searcher.index(self.sample_tools)
        
        queries = [
            "create@#$%",
            "send/message",
            "sql_query()",
            "repo:owner/name"
        ]
        
        for query in queries:
            try:
                results = self.searcher.search(query)
                self.assertIsInstance(results, list)
            except Exception as e:
                self.fail(f"Query '{query}' raised exception: {e}")
    
    def test_unicode_query(self):
        """Test search with unicode characters"""
        self.searcher.index(self.sample_tools)
        
        queries = [
            "创建问题",  # Chinese
            "создать задачу",  # Russian
            "créer un problème"  # French
        ]
        
        for query in queries:
            try:
                results = self.searcher.search(query)
                self.assertIsInstance(results, list)
            except Exception as e:
                self.fail(f"Unicode query '{query}' raised exception: {e}")
    
    def test_very_long_query(self):
        """Test with very long query (1000+ chars)"""
        self.searcher.index(self.sample_tools)
        
        long_query = "test " * 1000
        try:
            results = self.searcher.search(long_query)
            self.assertIsInstance(results, list)
        except Exception as e:
            self.fail(f"Long query raised exception: {e}")
    
    def test_negative_top_k(self):
        """Test with invalid top_k value"""
        self.searcher.index(self.sample_tools)
        
        results = self.searcher.search("test", top_k=-1)
        # Should handle gracefully (maybe return default or empty)
        self.assertIsInstance(results, list)


class TestMockedComponents(unittest.TestCase):
    """Tests with mocked external dependencies"""
    
    def setUp(self):
        self.sample_tools = [
            {
                "name": "create_github_issue",
                "description": "Create a new issue in a GitHub repository",
                "parameters": {
                    "repo": {"type": "string", "description": "Repository name"},
                    "title": {"type": "string", "description": "Issue title"},
                    "body": {"type": "string", "description": "Issue description"}
                },
                "required": ["repo", "title"]
            },
            {
                "name": "send_slack_message",
                "description": "Send a message to a Slack channel",
                "parameters": {
                    "channel": {"type": "string", "description": "Channel ID"},
                    "text": {"type": "string", "description": "Message text"}
                },
                "required": ["channel", "text"]
            },
            {
                "name": "execute_sql_query",
                "description": "Execute a SQL query on the database",
                "parameters": {
                    "query": {"type": "string", "description": "SQL query"},
                    "database": {"type": "string", "description": "Database name"}
                },
                "required": ["query"]
            }
        ]
    
    @patch('src.hybrid_searcher.CodeEmbedder')
    @patch('src.hybrid_searcher.LocalReranker')
    def test_mocked_embedder(self, mock_reranker, mock_embedder):
        """Test with mocked embedder"""
        # Setup mocks
        mock_embedder_instance = Mock()
        mock_embedder_instance.batch_encode_tools.return_value = [[0.1] * 1536] * 3
        mock_embedder_instance.encode_query.return_value = [0.1] * 1536
        mock_embedder.return_value = mock_embedder_instance
        
        mock_reranker_instance = Mock()
        mock_reranker_instance.rerank.return_value = [
            {'index': 0, 'relevance_score': 0.9, 'document': 'test'}
        ]
        mock_reranker.return_value = mock_reranker_instance
        
        # Create searcher with mocks
        searcher = HybridToolSearcher()
        searcher.index(self.sample_tools)
        
        results = searcher.search("test")
        self.assertEqual(len(results), 1)


def run_tests():
    """Run all tests"""
    # Create test loader
    loader = unittest.TestLoader()
    
    # Create test suite
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestHybridToolSearcher))
    suite.addTests(loader.loadTestsFromTestCase(TestConfig))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)