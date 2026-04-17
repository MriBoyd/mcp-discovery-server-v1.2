#!/usr/bin/env python3
import sys
import os
import subprocess

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    manager_path = os.path.join(base, "mcp_registry_manager.py")
    
    # Forward all arguments and set mode to config
    cmd = [sys.executable, manager_path, "--mode", "config"] + sys.argv[1:]
    
    # Ensure --output defaults to mcp_servers.json
    if "--output" not in sys.argv:
        cmd.extend(["--output", "mcp_servers.json"])
        
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
