import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import torch
import os

# Mock Config before importing CodeEmbedder
with patch("src.config.Config") as mock_config:
    mock_config.DEVICE = "cpu"
    from src.code_embedder import CodeEmbedder

class TestCodeEmbedder(unittest.TestCase):
    
    @patch("src.code_embedder.AutoTokenizer.from_pretrained")
    @patch("src.code_embedder.AutoModel.from_pretrained")
    @patch("src.code_embedder.Path.exists")
    def setUp(self, mock_exists, mock_model_from_pretrained, mock_tokenizer_from_pretrained):
        mock_exists.return_value = True
        
        self.mock_tokenizer = MagicMock()
        mock_tokenizer_from_pretrained.return_value = self.mock_tokenizer
        
        self.mock_model = MagicMock()
        mock_model_from_pretrained.return_value = self.mock_model
        
        with patch("torch.quantization.quantize_dynamic") as mock_quantize:
            mock_quantize.return_value = self.mock_model
            self.embedder = CodeEmbedder(model_path="./dummy-model")

    def test_encode_single_text(self):
        # Setup mock tokenizer output
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        
        # Setup mock model output
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(1, 3, 896)
        self.mock_model.return_value = mock_output
        self.mock_model.device = "cpu"

        # Patch F.normalize to return a predictable value
        with patch("torch.nn.functional.normalize") as mock_normalize:
            mock_normalize.return_value = torch.ones(1, 896)
            
            embeddings = self.embedder.encode("test query")
            
            self.assertEqual(len(embeddings), 1)
            self.assertEqual(len(embeddings[0]), 896)
            self.mock_tokenizer.assert_called_once()
            self.mock_model.assert_called_once()

    def test_encode_batch(self):
        texts = ["query 1", "query 2"]
        self.mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2], [3, 4]]),
            "attention_mask": torch.tensor([[1, 1], [1, 1]])
        }
        
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 2, 896)
        self.mock_model.return_value = mock_output
        self.mock_model.device = "cpu"

        with patch("torch.nn.functional.normalize") as mock_normalize:
            mock_normalize.return_value = torch.ones(2, 896)
            
            embeddings = self.embedder.encode(texts)
            
            self.assertEqual(len(embeddings), 2)
            self.assertEqual(len(embeddings[0]), 896)

    def test_last_token_pool(self):
        last_hidden_states = torch.tensor([
            [[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]], # seq len 3
            [[0.4, 0.4], [0.5, 0.5], [0.0, 0.0]]  # seq len 2 + padding
        ])
        attention_mask = torch.tensor([
            [1, 1, 1],
            [1, 1, 0]
        ])
        
        pooled = self.embedder.last_token_pool(last_hidden_states, attention_mask)
        
        # For first item, last token is at index 2: [0.3, 0.3]
        # For second item, last token is at index 1: [0.5, 0.5]
        self.assertTrue(torch.allclose(pooled[0], torch.tensor([0.3, 0.3])))
        self.assertTrue(torch.allclose(pooled[1], torch.tensor([0.5, 0.5])))

    def test_encode_query_caching(self):
        with patch.object(self.embedder, "encode", return_value=[[0.1]*896]) as mock_encode:
            # First call
            self.embedder.encode_query("cached query")
            # Second call
            self.embedder.encode_query("cached query")
            
            # Should only call encode once due to lru_cache
            self.assertEqual(mock_encode.call_count, 1)

    def test_batch_encode_tools(self):
        tool_texts = ["tool1", "tool2", "tool3"]
        with patch.object(self.embedder, "encode", side_effect=[
            [[0.1]*896, [0.2]*896], # batch 1
            [[0.3]*896]             # batch 2
        ]) as mock_encode:
            embeddings = self.embedder.batch_encode_tools(tool_texts, batch_size=2)
            
            self.assertEqual(len(embeddings), 3)
            self.assertEqual(mock_encode.call_count, 2)

if __name__ == "__main__":
    unittest.main()
