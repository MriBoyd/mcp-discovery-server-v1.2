import unittest
import os
from unittest.mock import patch
import importlib

class TestConfig(unittest.TestCase):
    
    def test_default_config(self):
        # Reload to ensure fresh state
        import src.config
        importlib.reload(src.config)
        from src.config import Config
        
        self.assertEqual(Config.EMBEDDING_DIM, 896)
        self.assertEqual(Config.BM25_CANDIDATES, 20)
        self.assertEqual(Config.FUSION_METHOD, "rrf")

    def test_env_override(self):
        env_vars = {
            "QDRANT_URL": "http://remote:6333",
            "MAX_OPEN_CONNECTIONS": "64",
            "GLOBAL_RATE": "50.0"
        }
        
        with patch.dict(os.environ, env_vars):
            import src.config
            importlib.reload(src.config)
            from src.config import Config
            
            self.assertEqual(Config.QDRANT_URL, "http://remote:6333")
            self.assertEqual(Config.MAX_OPEN_CONNECTIONS, 64)
            self.assertEqual(Config.GLOBAL_RATE, 50.0)

    def test_offline_mode_vars(self):
        import src.config
        importlib.reload(src.config)
        
        self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
        self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")

if __name__ == "__main__":
    unittest.main()
