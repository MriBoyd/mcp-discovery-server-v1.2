import unittest
from unittest.mock import patch, MagicMock
import sys
import os

class TestExtractRegistry(unittest.TestCase):
    
    @patch("subprocess.run")
    def test_main_execution(self, mock_run):
        # We need to simulate running extract_registry.py as a script
        # Mock sys.argv
        with patch.object(sys, 'argv', ['extract_registry.py', '--name', 'test_server', '--transport', 'stdio', '--command', 'node']):
            import extract_registry
            import importlib
            importlib.reload(extract_registry)
            
            # The script calls subprocess.run with a list that includes the path to mcp_registry_manager.py
            # and our forwarded args
            mock_run.assert_called_once()
            args = mock_run.call_args[0][0]
            self.assertIn("mcp_registry_manager.py", args[1])
            self.assertIn("--name", args)
            self.assertIn("test_server", args)
            self.assertIn("--output", args)
            self.assertIn("mcp_servers.json", args)

if __name__ == "__main__":
    unittest.main()
