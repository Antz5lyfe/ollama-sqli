import asyncio
import getpass
import os
import sqlite3
import warnings

import nest_asyncio
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.playwright.utils import \
    create_async_playwright_browser
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph_supervisor import create_supervisor
from pydantic import BaseModel, Field

from tools.all_tools import requests_tools

nest_asyncio.apply()


load_dotenv()

async def main():
    tools = requests_tools
    for tool in tools:
        print(f"Tool: {tool.name}, Description: {tool.description}")

    agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        prompt="Submit a POST request to the URL provided in form encoded data. The requests should contain a \'sentence\' key which contains a random sentence about cats",
        name="requests_agent",
        tools=tools
    )
    resp = await agent.ainvoke({ "messages": [HumanMessage(content="https://webhook.site/9f54d692-cbc4-43f2-94e0-eacc971b0870")] })
    print(resp)


if __name__ == "__main__":
    asyncio.run(main())  