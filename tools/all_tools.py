from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_core.tools import Tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentStateWithStructuredResponse

from mcp_client import get_mcp_tools
from playwright_tools.custom_playwright_toolkit import PlayWrightBrowserToolkit
from operator import add


class PentestState(AgentStateWithStructuredResponse):
    tries: int
    should_terminate: bool
    reason: str
    url: str
    attempts: Annotated[list[dict[str, str]], add]


search = GoogleSerperAPIWrapper()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Use this to search the web for information",
)


def playwright_tools():
    async_browser = create_async_playwright_browser(headless=False)  # headful mode
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    return toolkit.get_tools()


def rag(urls: list[str], name: str, description: str):
    docs = [WebBaseLoader(url).load() for url in urls]

    docs_list = [item for sublist in docs for item in sublist]

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=100, chunk_overlap=50
    )
    doc_splits = text_splitter.split_documents(docs_list)

    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits, embedding=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(retriever, name, description)
    return retriever_tool


sqli_rag_tool = rag(
    [
        "https://book.hacktricks.wiki/en/pentesting-web/sql-injection/index.html",
        "https://book.hacktricks.wiki/en/pentesting-web/sql-injection/postgresql-injection/index.html",
        "https://book.hacktricks.wiki/en/pentesting-web/sql-injection/mysql-injection/index.html",
        "https://raw.githubusercontent.com/payloadbox/sql-injection-payload-list/refs/heads/master/README.md",
        "https://www.invicti.com/blog/web-security/sql-injection-cheat-sheet/",
        "https://www.cobalt.io/blog/a-pentesters-guide-to-sql-injection-sqli",
        "https://github.com/AdmiralGaust/SQL-Injection-cheat-sheet",
        "https://portswigger.net/web-security/sql-injection/cheat-sheet",
        "https://portswigger.net/web-security/sql-injection/union-attacks",
        "https://github.com/swisskyrepo/PayloadsAllTheThings/blob/master/SQL%20Injection/README.md",
    ],
    "retrieve_sqli_information",
    "Search and return information about SQL Injection and payloads from SQL Injection Cheat Sheets.",
)

ffuf_rag_tool = rag(
    [
        "https://github.com/ffuf/ffuf/wiki",
        "https://medium.com/quiknapp/fuzz-faster-with-ffuf-c18c031fc480",
        "https://www.freecodecamp.org/news/web-security-fuzz-web-applications-using-ffuf/",
        "https://medium.com/@NiaziSec/mastering-ffuf-the-full-toolkit-3e7266dcced9",
    ],
    "retrieve_ffuf_information",
    "Search and return information about FFUF usage",
)

requests_tools = RequestsToolkit(
    requests_wrapper=TextRequestsWrapper(headers={}),
    allow_dangerous_requests=True,
).get_tools()

file_management_tools = FileManagementToolkit(
    root_dir=str("sandbox"),
    selected_tools=["read_file", "list_directory", "file_search"],
).get_tools()


# async def all_tools():
#     return (
#         (await get_mcp_tools())
#         + [search_tool, sqli_rag_tool, ffuf_rag_tool]
#         + playwright_tools()
#         + file_management_tools
#     )


@tool
def get_attempts(state: Annotated[PentestState, InjectedState]) -> int:
    """
    Returns the number of attempts made by the Pentest Agents.
    """
    return state["tries"]


async def scanner_tools():
    return (
        (await get_mcp_tools("scanner_mcp.json"))
        + [search_tool]
        + playwright_tools()
        + file_management_tools
    )


async def planner_tools():
    return (await get_mcp_tools("planner_mcp.json")) + [search_tool, sqli_rag_tool]


def attacker_tools():
    return playwright_tools() + requests_tools
