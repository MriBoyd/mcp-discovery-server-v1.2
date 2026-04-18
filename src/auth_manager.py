import json
import os
import time
import logging
import httpx
import base64
import hashlib
import secrets
from pathlib import Path
from typing import Optional, Dict, Tuple

class AuthManager:
    """Manages OAuth tokens for MCP servers indexed by Ansam"""
    
    def __init__(self, storage_path: str = "tokens.json"):
        self.storage_path = Path(storage_path)
        self.tokens: Dict[str, Dict] = self._load_tokens()

    def _load_tokens(self) -> Dict[str, Dict]:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logging.error(f"Failed to load tokens: {e}")
        return {}

    def _save_tokens(self):
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self.tokens, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save tokens: {e}")

    def get_token(self, server_name: str) -> Optional[str]:
        """Get a valid access token, refreshing is handled by the caller via refresh_token() if this returns None"""
        server_auth = self.tokens.get(server_name)
        if not server_auth:
            return None

        # Check if expired (with 1 min buffer)
        expires_at = server_auth.get("expires_at", 0)
        if time.time() < expires_at - 60:
            return server_auth.get("access_token")

        return None

    async def refresh_token(self, server_name: str, server_config: Dict) -> Optional[str]:
        """Refresh an expired token using the refresh_token"""
        server_auth = self.tokens.get(server_name)
        if not server_auth or "refresh_token" not in server_auth:
            return None

        auth_config = server_config.get("auth", {})
        token_endpoint = auth_config.get("token_endpoint")
        
        if not token_endpoint:
            return None

        payload = {
            "grant_type": "refresh_token",
            "refresh_token": server_auth["refresh_token"],
            "client_id": auth_config.get("client_id"),
            "client_secret": auth_config.get("client_secret"),
        }

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(token_endpoint, data=payload)
                if response.status_code == 401 or response.status_code == 400:
                    # Refresh token likely revoked or expired
                    del self.tokens[server_name]
                    self._save_tokens()
                    return None
                    
                response.raise_for_status()
                new_data = response.json()
                
                # Merge with existing data to keep the refresh_token if not provided in response
                updated_auth = {**server_auth, **new_data}
                self.save_token_data(server_name, updated_auth)
                return updated_auth.get("access_token")
            except Exception as e:
                logging.error(f"Refresh failed for {server_name}: {e}")
                return None

    def save_token_data(self, server_name: str, token_data: Dict):
        """Save raw token response and calculate expiry"""
        # Calculate expiry
        expires_in = token_data.get("expires_in", 3600)
        token_data["expires_at"] = time.time() + expires_in
        
        self.tokens[server_name] = token_data
        self._save_tokens()

    def generate_pkce_pair(self) -> Tuple[str, str]:
        """Generate a PKCE code_verifier and code_challenge (S256)

        Returns (code_verifier, code_challenge)
        """
        # RFC7636: code_verifier length between 43 and 128
        verifier = secrets.token_urlsafe(64)
        # compute challenge
        m = hashlib.sha256()
        m.update(verifier.encode('utf-8'))
        digest = m.digest()
        challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode('ascii')
        return verifier, challenge

    def get_auth_url(self, server_name: str, server_config: Dict, redirect_uri: str, code_challenge: Optional[str] = None, code_challenge_method: Optional[str] = None) -> Optional[str]:
        """Generate the authorization URL for the user"""
        auth_config = server_config.get("auth", {})
        auth_endpoint = auth_config.get("authorization_endpoint")
        
        if not auth_endpoint:
            return None
            
        params = {
            "response_type": "code",
            "client_id": auth_config.get("client_id"),
            "redirect_uri": redirect_uri,
            "scope": " ".join(auth_config.get("scopes_supported", [])),
            "state": server_name # Using state to track which server we're authenticating
        }

        # Include PKCE parameters when provided
        if code_challenge:
            params["code_challenge"] = code_challenge
        if code_challenge_method:
            params["code_challenge_method"] = code_challenge_method
        
        # Simple URL builder (basic escaping)
        parts = []
        for k, v in params.items():
            if v is None:
                continue
            parts.append(f"{k}={v}")
        query = "&".join(parts)
        return f"{auth_endpoint}?{query}"

    async def exchange_code(self, server_name: str, server_config: Dict, code: str, redirect_uri: str, code_verifier: Optional[str] = None) -> Optional[str]:
        """Exchange auth code for access/refresh tokens"""
        auth_config = server_config.get("auth", {})
        token_endpoint = auth_config.get("token_endpoint")
        
        if not token_endpoint:
            return None

        payload = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "client_id": auth_config.get("client_id"),
            "client_secret": auth_config.get("client_secret"),
        }

        # Include PKCE code_verifier when present
        if code_verifier:
            payload["code_verifier"] = code_verifier

        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(token_endpoint, data=payload)
                response.raise_for_status()
                token_data = response.json()
                
                self.save_token_data(server_name, token_data)
                return token_data.get("access_token")
            except Exception as e:
                logging.error(f"Code exchange failed for {server_name}: {e}")
                return None
