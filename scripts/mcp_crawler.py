#!/usr/bin/env python3
import sys
import os
import subprocess

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    manager_path = os.path.join(base, "../src/mcp_registry_manager.py")
    
    # Forward all arguments but set mode to flat
    cmd = [sys.executable, manager_path, "--mode", "flat"] + sys.argv[1:]
    
    # Ensure --output is specified or defaults to registry_dump.json
    if "--output" not in sys.argv:
        cmd.extend(["--output", "registry_dump.json"])
        
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
