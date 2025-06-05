from langchain_openai import ChatOpenAI
from mcp.client.stdio import stdio_client
from mcp import ClientSession, StdioServerParameters
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from dotenv import load_dotenv
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
import json

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

def load_mcp_servers_from_json(json_path="mcp.json"):
    with open(json_path, "r") as f:
        data = json.load(f)
    servers = {}
    for entry in data.get("servers", []):
        name = entry["key"]
        params = entry["params"].copy()
        params["transport"] = "stdio"
        servers[name] = params
    return servers

async def get_mcp_tools():
  client = MultiServerMCPClient(load_mcp_servers_from_json())
  # try:
  tools = await client.get_tools()
  # except Exception as e:
  #   print(f"Error loading tools from MCP server: {e}")
  #   return
  print("Loaded tools from MCP server")
  for tool in tools:
    print(f"Tool: {tool.name}, Description: {tool.description}")
  
  return tools
  # agent = create_react_agent(model, tools)
  # agent_response = await agent.ainvoke({"messages": "what's (4 + 6) x 14?"})
  # print(agent_response["messages"][-1].content)
  # return agent_response["messages"][-1].content

# if __name__ == "__main__":
#   asyncio.run(ge())