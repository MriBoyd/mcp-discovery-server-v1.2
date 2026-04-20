import unittest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from src.mcp_registry_manager import MCPRegistryManager

class TestMCPRegistryManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.manager = MCPRegistryManager(timeout=1, debug=False)

    def test_parse_header_entries(self):
        headers = ["Content-Type: application/json", "Authorization: Bearer token"]
        parsed = self.manager.parse_header_entries(headers)
        self.assertEqual(parsed["Content-Type"], "application/json")
        self.assertEqual(parsed["Authorization"], "Bearer token")

    def test_parse_tool_for_registry(self):
        mock_tool = MagicMock()
        mock_tool.model_dump.return_value = {"name": "test_tool", "description": "desc"}
        
        parsed = self.manager.parse_tool_for_registry(mock_tool, "server1", "stdio")
        self.assertEqual(parsed["name"], "test_tool")
        self.assertEqual(parsed["server_origin"], "server1")
        self.assertEqual(parsed["transport_used"], "stdio")

    def test_parse_tool_for_config(self):
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"
        mock_tool.description = "desc"
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"arg1": {"type": "string", "description": "d1"}},
            "required": ["arg1"]
        }
        
        parsed = self.manager.parse_tool_for_config(mock_tool)
        self.assertEqual(parsed["name"], "test_tool")
        self.assertEqual(parsed["parameters"]["arg1"]["required"], True)

    @patch("src.mcp_registry_manager.stdio_client")
    @patch("src.mcp_registry_manager.ClientSession")
    async def test_extract_from_stdio_server(self, mock_session_class, mock_stdio_client):
        # Mock stdio_client context manager
        mock_read = MagicMock()
        mock_write = MagicMock()
        mock_stdio_client.return_value.__aenter__.return_value = (mock_read, mock_write)
        
        # Mock ClientSession
        mock_session = AsyncMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session
        
        mock_response = MagicMock()
        mock_response.tools = [MagicMock(name="tool1")]
        mock_session.list_tools.return_value = mock_response

        config = {"transport": "stdio", "command": "ls", "args": ["-l"]}
        tools = await self.manager.extract_from_server("test_server", config)
        
        self.assertEqual(len(tools), 1)
        mock_session.initialize.assert_awaited_once()
        mock_session.list_tools.assert_awaited_once()

    def test_save_as_flat_list(self):
        tools = [{"name": "t1", "server_origin": "s1"}]
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            self.manager.save_as_flat_list(tools, tmp_path)
            with open(tmp_path, 'r') as f:
                saved = json.load(f)
                self.assertEqual(len(saved), 1)
                self.assertEqual(saved[0]["name"], "t1")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_save_as_server_config(self):
        servers = [{"name": "s1", "tools": [{"name": "t1"}]}]
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            self.manager.save_as_server_config(servers, tmp_path)
            with open(tmp_path, 'r') as f:
                saved = json.load(f)
                self.assertIn("servers", saved)
                self.assertEqual(saved["servers"][0]["name"], "s1")
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == "__main__":
    unittest.main()
