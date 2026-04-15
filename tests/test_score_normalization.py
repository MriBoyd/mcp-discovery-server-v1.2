# tests/test_score_normalization.py

import unittest
import numpy as np
from typing import List, Union
from src.hybrid_searcher import HybridToolSearcher

class TestScoreNormalization(unittest.TestCase):
    def setUp(self):
        self.searcher = HybridToolSearcher()
    
    def test_minmax_normalization(self):
        """Test min-max normalization"""
        # Convert to float explicitly
        scores: List[float] = [10.0, 20.0, 30.0, 40.0, 50.0]
        normalized = self.searcher._normalize_scores(scores)
        
        self.assertEqual(normalized[0], 0.0)
        self.assertEqual(normalized[4], 1.0)
        self.assertEqual(normalized[2], 0.5)
    
    def test_minmax_single_value(self):
        """Test min-max with single value"""
        scores: List[float] = [25.0]
        normalized = self.searcher._normalize_scores(scores)
        
        self.assertEqual(normalized[0], 0.5)  # Default neutral
    
    def test_softmax_normalization(self):
        """Test softmax normalization"""
        scores: List[float] = [1.0, 2.0, 3.0]
        normalized = self.searcher._normalize_scores_softmax(scores)
        
        # Should sum to 1
        self.assertAlmostEqual(sum(normalized), 1.0)
        
        # Higher scores get higher probability
        self.assertGreater(normalized[2], normalized[1])
        self.assertGreater(normalized[1], normalized[0])
    
    def test_rank_normalization(self):
        """Test rank-based normalization"""
        scores: List[float] = [100.0, 50.0, 25.0, 10.0]
        normalized = self.searcher._normalize_scores_rank(scores)
        
        # Best score gets 1.0, worst gets 0.0
        self.assertEqual(normalized[0], 1.0)  # Index 0 has highest score
        self.assertEqual(normalized[3], 0.0)  # Index 3 has lowest score