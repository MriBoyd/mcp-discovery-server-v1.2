import asyncio

from mirascope import llm

from mcp.client.stdio import StdioServerParameters

async def main():
    # Define multiple stdio server configurations
    server_configs = [
        StdioServerParameters(
            command="node",
            args=["/mnt/0BF70EDD0BF70EDD/hackernews-mcp/build/index.js"],
        ),
        StdioServerParameters(
            command="uv",
            args=["run", "/mnt/0BF70EDD0BF70EDD/gmail-mcp/src/gmail/server.py", "--creds-file-path", "/mnt/0BF70EDD0BF70EDD/gmail-mcp/credentials.json", "--token-path", "/mnt/0BF70EDD0BF70EDD/gmail-mcp/token.json"],
            env=None
        ),
        # Add more servers as needed
    ]
    
    # Collect tools from all stdio clients
    all_tools = []
    
    for server_params in server_configs:
        async with llm.mcp.stdio_client(server_params) as client:
            tools = await client.list_tools()
            all_tools.extend(tools)
    
    @llm.call("ollama/qwen3.5:4b", tools=all_tools)
    async def assistant(query: str):
        return query
    
    response = await assistant("send greeting message to mriboyd1240@gmail.com ")
    
    while response.tool_calls:
        tool_outputs = await response.execute_tools()
        response = await response.resume(tool_outputs)
        
        print("Tool calls executed, resuming assistant...")
        print("Current response:", response.text)
        print("Tool calls:", response.tool_calls)
    
    print(response.pretty())


asyncio.run(main())