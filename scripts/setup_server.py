import json
import asyncio
import argparse
import threading
import queue
import http.server
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from src.auth_manager import AuthManager

async def setup_server():
    parser = argparse.ArgumentParser(description="Ansam Server Setup & OAuth Authenticator")
    parser.add_argument("--name", required=True, help="Name of the MCP server (must match mcp_servers.json)")
    parser.add_argument("--callback", default="https://ansam.dev/oauth/callback", help="Redirect URI")
    parser.add_argument("--auto", action="store_true", help="Automatically open browser and accept token via local HTTP listener")
    parser.add_argument("--port", type=int, default=8000, help="Local listener port when using --auto")
    args = parser.parse_args()

    auth_manager = AuthManager()
    config_path = Path("src/mcp_servers.json")
    
    with open(config_path) as f:
        config = json.load(f)
    
    server_config = next((s for s in config["servers"] if s["name"] == args.name), None)
    if not server_config:
        print(f"❌ Server '{args.name}' not found in src/mcp_servers.json")
        return

    if "auth" not in server_config:
        print(f"ℹ️ Server '{args.name}' does not require OAuth. You're all set!")
        return

    # Generate Link and either accept code manually or start local listener
    if args.auto:
        port = args.port
        code_queue = queue.Queue()

        class _Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                parsed = urlparse(self.path)
                qs = parse_qs(parsed.query)
                code = qs.get('code', [None])[0]
                if code:
                    try:
                        code_queue.put_nowait(code)
                    except Exception:
                        pass

                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Authentication received. You can close this tab.</h1></body></html>")

            def log_message(self, format, *args):
                return

        httpd = http.server.HTTPServer(('127.0.0.1', port), _Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        callback_url = f"http://127.0.0.1:{port}/callback"
        # PKCE
        code_verifier, code_challenge = auth_manager.generate_pkce_pair()
        url = auth_manager.get_auth_url(args.name, server_config, callback_url, code_challenge=code_challenge, code_challenge_method="S256")
        print(f"\n--- 🔐 OAuth Setup (auto) for {args.name} ---")
        print(f"Opening browser to: {url}\n")
        webbrowser.open(url)

        try:
            code = code_queue.get(timeout=300)
        except queue.Empty:
            print("❌ Timeout waiting for authorization code. Try again or use the manual flow.")
            httpd.shutdown()
            return

        httpd.shutdown()
        print("\n3. Exchanging code for tokens...")
        token = await auth_manager.exchange_code(args.name, server_config, code, callback_url, code_verifier=code_verifier)
    else:
        # Manual paste flow
        # PKCE
        code_verifier, code_challenge = auth_manager.generate_pkce_pair()
        url = auth_manager.get_auth_url(args.name, server_config, args.callback, code_challenge=code_challenge, code_challenge_method="S256")
        print(f"\n--- 🔐 OAuth Setup for {args.name} ---")
        print(f"1. Open this URL in your browser:\n\n{url}\n")
        code = input("2. Paste the 'code' parameter from the redirect URL here: ").strip()
        print("\n3. Exchanging code for tokens...")
        token = await auth_manager.exchange_code(args.name, server_config, code, args.callback, code_verifier=code_verifier)
    
    if token:
        print(f"✅ Success! Tokens for '{args.name}' are now securely saved in tokens.json.")
        print("Ansam will now automatically handle authentication for this server.")
    else:
        print("❌ Failed to exchange code. Please check your credentials and try again.")

if __name__ == "__main__":
    asyncio.run(setup_server())
