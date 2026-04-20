import unittest
import json
import os
import time
import tempfile
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
from src.auth_manager import AuthManager

class TestAuthManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.storage_path = os.path.join(self.temp_dir.name, "tokens.json")
        self.auth_manager = AuthManager(storage_path=self.storage_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_tokens_empty(self):
        self.assertEqual(self.auth_manager.tokens, {})

    def test_save_and_load_tokens(self):
        token_data = {"access_token": "test_token", "expires_at": time.time() + 3600}
        self.auth_manager.save_token_data("test_server", token_data)
        
        # New instance to test loading from file
        new_manager = AuthManager(storage_path=self.storage_path)
        self.assertIn("test_server", new_manager.tokens)
        self.assertEqual(new_manager.tokens["test_server"]["access_token"], "test_token")

    def test_get_token_valid(self):
        expires_at = time.time() + 3600
        self.auth_manager.tokens = {
            "test_server": {"access_token": "valid_token", "expires_at": expires_at}
        }
        self.assertEqual(self.auth_manager.get_token("test_server"), "valid_token")

    def test_get_token_expired(self):
        expires_at = time.time() - 3600
        self.auth_manager.tokens = {
            "test_server": {"access_token": "expired_token", "expires_at": expires_at}
        }
        self.assertIsNone(self.auth_manager.get_token("test_server"))

    def test_get_token_buffer(self):
        # Within 1 minute buffer
        expires_at = time.time() + 30 
        self.auth_manager.tokens = {
            "test_server": {"access_token": "buffering_token", "expires_at": expires_at}
        }
        self.assertIsNone(self.auth_manager.get_token("test_server"))

    @patch("httpx.AsyncClient.post")
    async def test_refresh_token_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "new_token", "expires_in": 3600}
        mock_post.return_value = mock_response

        self.auth_manager.tokens = {
            "test_server": {"refresh_token": "ref_token"}
        }
        server_config = {
            "auth": {
                "token_endpoint": "http://example.com/token",
                "client_id": "id",
                "client_secret": "secret"
            }
        }
        
        token = await self.auth_manager.refresh_token("test_server", server_config)
        self.assertEqual(token, "new_token")
        self.assertEqual(self.auth_manager.tokens["test_server"]["access_token"], "new_token")

    @patch("httpx.AsyncClient.post")
    async def test_refresh_token_failure(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_post.return_value = mock_response

        self.auth_manager.tokens = {
            "test_server": {"refresh_token": "ref_token"}
        }
        server_config = {"auth": {"token_endpoint": "http://example.com/token"}}
        
        token = await self.auth_manager.refresh_token("test_server", server_config)
        self.assertIsNone(token)
        self.assertNotIn("test_server", self.auth_manager.tokens)

    def test_generate_pkce_pair(self):
        verifier, challenge = self.auth_manager.generate_pkce_pair()
        self.assertTrue(len(verifier) >= 43)
        self.assertTrue(len(challenge) > 0)

    def test_get_auth_url(self):
        server_config = {
            "auth": {
                "authorization_endpoint": "http://example.com/auth",
                "client_id": "id",
                "scopes_supported": ["read", "write"]
            }
        }
        url = self.auth_manager.get_auth_url(
            "test_server", 
            server_config, 
            "http://redirect", 
            code_challenge="challenge", 
            code_challenge_method="S256"
        )
        self.assertIn("response_type=code", url)
        self.assertIn("client_id=id", url)
        self.assertIn("code_challenge=challenge", url)
        self.assertIn("state=test_server", url)

    @patch("httpx.AsyncClient.post")
    async def test_exchange_code(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"access_token": "exchanged_token", "expires_in": 3600}
        mock_post.return_value = mock_response

        server_config = {"auth": {"token_endpoint": "http://example.com/token"}}
        token = await self.auth_manager.exchange_code("test_server", server_config, "code", "http://redirect")
        
        self.assertEqual(token, "exchanged_token")
        self.assertEqual(self.auth_manager.tokens["test_server"]["access_token"], "exchanged_token")

if __name__ == "__main__":
    unittest.main()
