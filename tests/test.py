import asyncio
from contextlib import AsyncExitStack

from mirascope import llm

from mcp.client.stdio import StdioServerParameters

async def main():
    # Define multiple stdio server configurations
    server_configs = [
        StdioServerParameters(
            command="node",
            args=["/mnt/0BF70EDD0BF70EDD/hackernews-mcp/build/index.js"],
        ),
        # Add more servers as needed
    ]
    
    # Collect tools from all stdio clients
    
    async with AsyncExitStack() as stack:
        all_tools = []
        
        client =  await stack.enter_async_context(llm.mcp.sse_client("http://localhost:8000/sse"))
        all_tools.extend(await client.list_tools()) 
        
        for server_params in server_configs:
            client =  await stack.enter_async_context(llm.mcp.stdio_client(server_params))
            tools = await client.list_tools()
            all_tools.extend(tools)
        
        @llm.call("ollama/qwen3.5:4b", tools=all_tools)
        async def assistant(query: str):
            return [llm.messages.system("You are a helpful assistant. My email is mriboyd1240@gmail.com, always search for tools related to user queries if you do not have them available. and use them respond to call call_tool"), llm.messages.user(query)]
    
        
        response = await assistant("send random email to me")
        
        while response.tool_calls:
            tool_outputs = await response.execute_tools()
            response = await response.resume(tool_outputs)
            
            print("Tool calls executed, resuming assistant...")
            print("Current response:", response.text)
            print("Tool calls:", response.tool_calls)
        
        print(response.pretty())


asyncio.run(main())