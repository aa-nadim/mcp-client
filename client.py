import asyncio
import sys
import json
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # load environment variables from .env

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        
        # Initialize OpenAI client with API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not found. Please check your .env file.")
        self.openai_client = OpenAI(api_key=api_key)

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
            
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
            
            await self.session.initialize()
            
            # List available tools
            response = await self.session.list_tools()
            tools = response.tools
            print("\nConnected to server with tools:", [tool.name for tool in tools])
            return True
        except Exception as e:
            print(f"Failed to connect to server: {str(e)}")
            return False

    async def process_query(self, query: str) -> str:
        """Process a query using OpenAI and available tools"""
        if not self.session:
            return "Error: Not connected to MCP server"
            
        messages = [
            {
                "role": "user",
                "content": query
            }
        ]

        response = await self.session.list_tools()
        available_tools = [{ 
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.inputSchema
            }
        } for tool in response.tools]

        # Initial OpenAI API call
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",  # or another model you prefer
            messages=messages,
            tools=available_tools,
            tool_choice="auto"
        )

        # Process response and handle tool calls
        final_text = []
        response_message = response.choices[0].message

        if response_message.content:
            final_text.append(response_message.content)
        
        # Handle tool calls if present
        if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = tool_call.function.arguments
                
                # Convert string args to dict if necessary
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse tool arguments: {tool_args}")
                        continue
                
                # Execute tool call
                try:
                    print(f"Calling tool {tool_name} with args {tool_args}")
                    result = await self.session.call_tool(tool_name, tool_args)
                    final_text.append(f"[Tool result from {tool_name}: {result.content}]")
                    
                    # Continue conversation with tool results
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": tool_call.id,
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "arguments": tool_call.function.arguments
                                }
                            }
                        ]
                    })
                    
                    # Add tool results to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result.content
                    })
                except Exception as e:
                    final_text.append(f"[Error calling tool {tool_name}: {str(e)}]")
                    continue
                
            # Get next response from OpenAI
            try:
                second_response = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages
                )
                
                if second_response.choices[0].message.content:
                    final_text.append(second_response.choices[0].message.content)
            except Exception as e:
                final_text.append(f"[Error getting final response: {str(e)}]")

        return "\n".join(final_text)

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        while True:
            try:
                query = input("\nQuery: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not query:
                    continue
                    
                response = await self.process_query(query)
                print("\n" + response)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        connected = await client.connect_to_server(sys.argv[1])
        if connected:
            await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())