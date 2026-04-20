import unittest
from unittest.mock import patch, MagicMock
import torch

# Mock Config before importing LocalReranker
with patch("src.config.Config") as mock_config:
    mock_config.DEVICE = "cpu"
    mock_config.FINAL_RESULTS = 3
    from src.local_reranker import LocalReranker

class TestLocalReranker(unittest.TestCase):
    
    @patch("src.local_reranker.AutoTokenizer.from_pretrained")
    @patch("src.local_reranker.AutoModel.from_pretrained")
    def setUp(self, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        self.mock_tokenizer = MagicMock()
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        
        self.mock_model = MagicMock()
        mock_model_from_pretrained.return_value = self.mock_model
        
        with patch("torch.quantization.quantize_dynamic") as mock_quantize:
            mock_quantize.return_value = self.mock_model
            self.reranker = LocalReranker(model_path="./dummy-rerank")

    def test_rerank_success(self):
        query = "test query"
        documents = ["doc1", "doc2", "doc3"]
        
        # Setup mock rerank output
        self.mock_model.rerank.return_value = [
            {'index': 1, 'relevance_score': 0.9, 'document': 'doc2'},
            {'index': 0, 'relevance_score': 0.5, 'document': 'doc1'}
        ]

        results = self.reranker.rerank(query, documents, top_n=2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['index'], 1)
        self.assertEqual(results[0]['relevance_score'], 0.9)
        self.mock_model.rerank.assert_called_once_with(
            query=query, documents=documents, top_n=2
        )

    def test_rerank_empty_docs(self):
        results = self.reranker.rerank("query", [])
        self.assertEqual(results, [])

    def test_rerank_default_top_n(self):
        query = "test query"
        documents = ["doc1"]
        self.mock_model.rerank.return_value = [{'index': 0, 'relevance_score': 0.8, 'document': 'doc1'}]
        
        self.reranker.rerank(query, documents)
        
        # Should use Config.FINAL_RESULTS (mocked as 3)
        self.mock_model.rerank.assert_called_once_with(
            query=query, documents=documents, top_n=3
        )

if __name__ == "__main__":
    unittest.main()
