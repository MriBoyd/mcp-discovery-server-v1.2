from mirascope.core import openai, prompt_template
from pydantic import BaseModel, Field
import json
import asyncio

# This script simulates the LLM's logic using Mirascope and our Discovery Server
# In a real scenario, this agent would be initialized with access to the Discovery Server tools.

class DiscoveryAgent:
    def __init__(self, discovery_client):
        self.client = discovery_client
        # We start with only the discovery tools in context
        self.available_tools = [
            "discover_tools", 
            "get_tool_definition", 
            "call_mcp_tool", 
            "recommend_next_tools"
        ]

    @openai.call(model="gpt-4o")
    @prompt_template(
        """
        SYSTEM:
        You are an advanced agent with dynamic tool discovery capabilities.
        You do NOT have all tools loaded by default to prevent context pollution.
        
        STEP 1: Use `discover_tools` to find relevant tools for the user request.
        STEP 2: Use `get_tool_definition` to load the full schema of the tools you choose.
        STEP 3: Use `call_mcp_tool` to execute them.
        STEP 4: Use `recommend_next_tools` if you need hints for the next step.

        USER: {query}
        """
    )
    async def run(self, query: str):
        # Mirascope handles tool calling internally if we pass them
        # For this demo, we'll just show the prompt structure
        ...

# ---------------------------------------------------------
# Mock Simulation of the Loop the user described:
# "see weather, research, send telegram, send email"
# ---------------------------------------------------------

async def main():
    print("🚀 Starting Agentic Tool Discovery Loop...")
    
    query = "Check weather today in London, research if it's good for a BBQ, and send results to my Telegram saved messages and email my friends."
    
    print(f"\n[User Query]: {query}")
    
    # 1. LLM calls discover_tools(query="weather, research, telegram, email")
    # Returns: weather_get_forecast, web_search_search, telegram_send_message, email_send_email
    
    # 2. LLM calls get_tool_definition(tool_id="weather_get_forecast")
    # Returns: { parameters: { location: string, ... } }
    
    # 3. LLM calls call_mcp_tool(tool_id="weather_get_forecast", arguments={"location": "London"})
    # Returns: { result: "Sunny, 22C" }
    
    # 4. LLM calls recommend_next_tools(last_tool_id="weather_get_forecast")
    # Returns: [ {id: "web_search_search", reason: "Research if it's good for BBQ"} ]
    
    # ... and so on.
    
    print("\n✅ Discovery Toolset is ready in 'server.py'.")
    print("✅ Reranker is optimized in 'reranker.py'.")
    print("✅ Hybrid Index (BM25 + Vector) is in 'index.py'.")
    print("✅ Sample tools are in 'data/tools.json'.")

if __name__ == "__main__":
    asyncio.run(main())
