import asyncio
import getpass
import os
import sqlite3
import warnings

import nest_asyncio
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools.retriever import create_retriever_tool
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

from mcp_client import get_mcp_tools
from playwright_tools.custom_playwright_toolkit import PlayWrightBrowserToolkit

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
    # await page.goto("http://saturn.picoctf.net:55557/")
    # await page.click("input[name='username']")
    # from playwright.async_api import TimeoutError as PlaywrightTimeoutError
    # try:
    #     await page.fill(
    #             "input[name='username']",
    #             "' OR '1'='1",
    #             strict=False,
    #             timeout=1_000,
    #         )
    # except PlaywrightTimeoutError:
    #     return f"Unable to Fill on element"
    
    
    # await page.fill("input[name='username']", "testuser")
    # # await page.fill("input[name='password']", "testpass")
    # await page.click("input[type='submit']")
    # await page.wait_for_load_state("networkidle")
    # print(await page.content())


    agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        # prompt="Submit a form on the user provided website using playwright by entering a random username and password",
        name="form_agent",
        tools=tools
    )
    resp = await agent.ainvoke({ "messages": [HumanMessage(content="Go to https://www.google.com, search for Elon Musk and spaceX using the search textarea, click the google search button and return the summary of results you get. Use the fill tool to fill in fields and print out the url at each step.")]})
    # resp = await agent.ainvoke({ "messages": [HumanMessage(content="http://saturn.picoctf.net:55557/")] })
    print(resp)


if __name__ == "__main__":
    asyncio.run(main())  