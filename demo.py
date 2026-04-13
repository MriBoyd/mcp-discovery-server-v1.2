# demo.py
import json
from src.hybrid_searcher import HybridToolSearcher

def main():
    # Load your tools
    with open('tools/sample_tools.json', 'r') as f:
        data = json.load(f)
        tools = data if isinstance(data, list) else data.get('tools', [])
    
    # Initialize searcher
    searcher = HybridToolSearcher()
    
    # Index tools (first time only)
    searcher.index(tools)
    
    # Interactive search
    print("\n🔍 Offline Hybrid Tool Search")
    print("Models: Jina Code Embeddings 1.5B + Jina Reranker v3")
    print("=" * 60)
    
    while True:
        query = input("\n💬 Query (or 'quit'): ").strip()
        if query.lower() in ['quit', 'exit']:
            break
        
        if not query:
            continue
        
        results = searcher.search(query)
        
        print(f"\n📋 Top {len(results)} Results:\n")
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['tool_name']}")
            print(f"   Score: {result['relevance_score']:.4f}")
            print(f"   {result['tool_description'][:100]}...")
            print()

if __name__ == "__main__":
    main()