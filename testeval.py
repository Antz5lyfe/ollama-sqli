import getpass
import os
from typing import TypedDict
import warnings
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage
from pytz import all_timezones_set
from mcp_client import get_mcp_tools
import asyncio
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import Tool
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt.chat_agent_executor import (
    AgentStateWithStructuredResponse,
    AgentState,
)


class PentestState(AgentStateWithStructuredResponse):
    sentence: str


class ExploitEvaluatorOutput(TypedDict):
    """Random sentence"""

    sentence: str = Field(description="Radom sentence")


async def main(state: PentestState):
    exploit_evaluator_agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        prompt="Generate a random sentence.",
        response_format=("Generate a random sentence", ExploitEvaluatorOutput),
        name="random_sentence_generator",
        tools=[],
        state_schema=PentestState,
    )
    resp = await exploit_evaluator_agent.ainvoke(state)
    print(resp["structured_response"]["sentence"])
    return {
        "messages": [resp["messages"][-1]],
        "should_terminate": resp["structured_response"]["sentence"],
    }


asyncio.run(
    main(
        {
            "messages": [HumanMessage(content="Chen Ning")],
            "sentence": "",
            "structured_response": {},
        }
    )
)
