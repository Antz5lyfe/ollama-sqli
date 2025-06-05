import getpass
import os
import warnings
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from pydantic import BaseModel, Field
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage
from mcp_client import get_mcp_tools
import asyncio
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()


load_dotenv()

def playwright_tools():
    async_browser =  create_async_playwright_browser(headless=False)  # headful mode
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    return toolkit.get_tools(), async_browser

async def main():
    tools, async_browser = playwright_tools()
    for tool in tools:
        print(f"Tool: {tool.name}, Description: {tool.description}")
    # page = await async_browser.new_page()
    # await page.goto("http://saturn.picoctf.net:64743/")
    # await page.click("input[name='username']")
    # # await page.fill("input[name='username']", "testuser")
    # # await page.fill("input[name='password']", "testpass")
    # await page.click("input[type='submit']")
    # await page.wait_for_load_state("networkidle")
    # print(await page.content())


    # agent = create_react_agent(
    #     model="openai:gpt-4.1",
    #     prompt="Submit a form on the user provided website using playwright by entering a random username and password",
    #     name="form_agent",
    #     tools=tools
    # )
    # resp = await agent.ainvoke({ "messages": [HumanMessage(content="http://saturn.picoctf.net:64743/")] })
    # print(resp)


if __name__ == "__main__":
    asyncio.run(main())  