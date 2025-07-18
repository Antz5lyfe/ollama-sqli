from operator import add
from typing import Annotated, Union
import os
from langchain_community.vectorstores import Chroma
import json
from langchain_core.documents import Document

from langchain.tools.retriever import create_retriever_tool
from langchain_community.agent_toolkits import FileManagementToolkit
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_core.tools import Tool, tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import InjectedState
from langgraph.prebuilt.chat_agent_executor import AgentStateWithStructuredResponse
from langchain_ollama import OllamaEmbeddings

from mcp_client import get_mcp_tools
from playwright_tools.custom_playwright_toolkit import PlayWrightBrowserToolkit


class PentestState(AgentStateWithStructuredResponse):
    tries: int
    should_terminate: bool
    reason: str
    url: str
    attempts: list[dict[str, Union[dict, str]]]
    recommendation: dict
    successful_payload: Union[None, dict[str, str]]
    payloads: list


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

"""
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
"""

def rag(json_path: str, name: str, description: str):
    # Create a persistent directory for the vector store
    persist_directory = "vector_store"
    os.makedirs(persist_directory, exist_ok=True)
    
    print("Starting RAG initialization...")
    
    # Initialize embeddings
    print("Initializing Ollama embeddings...")
    embeddings = OllamaEmbeddings(
        model="qwen3:14b",
        base_url="http://localhost:11434"
    )
    print("Ollama embeddings initialized...")
    
    # Try to load existing vector store
    print("Checking for existing vector store...")
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    # If the store is empty, we need to create it
    if vectorstore._collection.count() == 0:
        print("Creating new vector store from local JSON...")

        # Load from local JSON file
        with open(json_path, "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        docs_list = [
            Document(page_content=doc["content"], metadata=doc.get("metadata", {}))
            for doc in raw_docs
        ]
        print(f"Loaded {len(docs_list)} documents from {json_path}")

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=100, chunk_overlap=50
        )
        doc_splits = text_splitter.split_documents(docs_list)
        print(f"Split into {len(doc_splits)} chunks...")

        # Process documents in smaller batches
        batch_size = 50
        for i in range(0, len(doc_splits), batch_size):
            batch = doc_splits[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1} of {(len(doc_splits) + batch_size - 1)//batch_size}...")
            vectorstore.add_documents(batch)
            
        # Persist the vector store
        vectorstore.persist()
        print("Vector store created and persisted...")
    else:
        print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
    
    retriever = vectorstore.as_retriever()
    retriever_tool = create_retriever_tool(retriever, name, description)
    print("RAG initialization complete!")
    return retriever_tool

"""
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
"""

# Offline RAG using local JSON
sqli_rag_tool = rag(
    json_path="sqli_docs.json",  # JSON file generated ahead of time
    name="retrieve_sqli_information",
    description="Search and return information about SQL Injection and payloads from SQL Injection Cheat Sheets.",
)

requests_tools = RequestsToolkit(
    requests_wrapper=TextRequestsWrapper(headers={}),
    allow_dangerous_requests=True,
).get_tools()

file_management_tools = FileManagementToolkit(
    root_dir=str("sandbox"),
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
        (await get_mcp_tools("scanner_mcp.json")) + [search_tool] + playwright_tools()
    )


async def planner_tools():
    return (await get_mcp_tools("planner_mcp.json")) + [search_tool, sqli_rag_tool]


def attacker_tools():
    return playwright_tools() + requests_tools


def report_writer_tools():
    return file_management_tools + [search_tool]
