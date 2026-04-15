
import os
import sys
import json
from typing import List, Dict, Any

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.hybrid_searcher import HybridToolSearcher
from src.config import Config

def test_capability_search():
    # Define some tools, including some with URL parameters
    tools = [
        {
            "name": "fetch_url_content",
            "description": "Fetch the content of a given URL",
            "parameters": {
                "url": {"type": "string", "description": "The URL to fetch"}
            },
            "required": ["url"]
        },
        {
            "name": "shorten_url",
            "description": "Shorten a long URL",
            "parameters": {
                "long_url": {"type": "string", "description": "The long URL to shorten"}
            },
            "required": ["long_url"]
        },
        {
            "name": "send_email",
            "description": "Send an email message",
            "parameters": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Email body"}
            },
            "required": ["to", "subject", "body"]
        },
        {
            "name": "calculate_sum",
            "description": "Calculate the sum of two numbers",
            "parameters": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        }
    ]

    searcher = HybridToolSearcher()
    print("Indexing tools (forcing re-index)...")
    searcher.index(tools, force_reindex=True)

    query = "tool that takes a URL"
    print(f"\nSearching for: '{query}'")
    results = searcher.search(query, top_k=5)

    print("\nResults:")
    for i, res in enumerate(results):
        print(f"{i+1}. {res['tool_name']} (Score: {res['relevance_score']:.4f})")
        # print(f"   Description: {res['tool_description']}")

if __name__ == "__main__":
    test_capability_search()
