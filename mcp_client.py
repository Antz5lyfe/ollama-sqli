import json

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
# from langchain_ollama import ChatOllama
from langchain_ollama.chat_models import ChatOllama


load_dotenv()

# model = ChatOpenAI(model="gpt-4o")
# model = ChatOllama(model="mistral:7b-instruct")
model=ChatOllama(model="qwen3:14b")


def load_mcp_servers_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    servers = {}
    for entry in data.get("servers", []):
        name = entry["key"]
        params = entry["params"].copy()
        params["transport"] = "stdio"
        servers[name] = params
    return servers


async def get_mcp_tools(json_path="mcp.json"):
    client = MultiServerMCPClient(load_mcp_servers_from_json(json_path))
    tools = await client.get_tools()
    print("Loaded tools from MCP server")
    for tool in tools:
        print(f"Tool: {tool.name}, Description: {tool.description}")

    return tools
