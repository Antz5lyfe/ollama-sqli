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
import nest_asyncio
nest_asyncio.apply()

warnings.filterwarnings("ignore", category=ResourceWarning)


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")



db_path = "memory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
sql_memory = SqliteSaver(conn)




scanner_agent_prompt = ("As the Scanner Agent, your mission is to perform a comprehensive reconnaissance of the target web application. Systematically identify and document all potential entry points for SQL Injection, including URLs, input fields, forms, query parameters, and authentication mechanisms. Use all available scanning tools to map the application's structure, enumerate endpoints, and detect possible vulnerabilities without executing any exploits or payloads. For each finding, provide detailed context (e.g., HTTP method, parameter names, sample values, and observed responses). Summarize your results in a structured format suitable for further analysis by the Planner Agent.")
planner_agent_prompt = ("As the Planner Agent, analyze the Scanner Agent’s findings to design a step-by-step exploitation strategy focused on SQL Injection. For each identified entry point, assess its potential for exploitation and prioritize targets based on likelihood of success and impact. Develop a tactical plan specifying which payloads, techniques, and tool configurations should be used by the Attacker Agent. You are provided with many tools, but  many are irrelevant for your job scope. Take note of past attempts, if any. Clearly justify your choices and include fallback options if initial attempts fail. Present your plan in a clear, actionable format for direct implementation.")
attacker_agent_prompt = ("As the Attacker Agent, your task is to execute the Planner Agent’s exploitation plan against the target web application. From the plan, choose one most promising payload to use, and perform that exploit. Take note of past attempts and do not repeat payloads. Do not perform any actions outside the provided plan. After each attempt, summarize the results and pass them to the Exploit Evaluator Agent for assessment. Note: The fetch tool is only used for performing get requests, if you want to get the contents of a page. To perform exploits, it is recommended to use playwright and manually enter the input fields and submit the form.")
exploit_evaluator_agent_prompt = ("As the Exploit Evaluator Agent, critically assess the outcomes of the Attacker Agent’s SQL Injection attempts. For each result, determine whether the exploit was successful by analyzing application responses, error messages, and any observable changes in behavior. Use browser tools and manual verification as needed. Clearly state whether each exploit succeeded or failed, and provide a concise explanation for your assessment. If unsuccessful, suggest possible reasons and improvements for future attempts.")
report_writer_agent_prompt = ("As the Report Writer Agent, compile a comprehensive and professional report detailing the entire pentesting operation. Summarize the objectives, methodology, and key findings, highlighting all successful SQL Injection exploits and their security implications. For each vulnerability, include technical details, evidence (such as payloads and responses), and recommended remediation steps. Ensure the report is clear, well-structured, and suitable for presentation to both technical and non-technical stakeholders.")
supervisor_agent_prompt = (
    "As the Supervisor of the pentesting AI app, your mission is to orchestrate the entire red teaming process, ensuring a strategic and efficient approach to uncovering vulnerabilities in CTF-style web applications. You are responsible for defining the overall plan and strategy, coordinating with other agents, and ensuring that each step aligns with the ultimate goal of identifying and exploiting security weaknesses. Your goal is to establish a comprehensive and adaptable strategy for the pentesting operation. Begin by outlining the objectives and key phases of the red teaming exercise. Set clear guidelines and priorities for the Scanner, Planner, Attacker, Exploit Evaluator, and Report Writer agents. The Planner, Attacker and Exploit Evaulator agents are under Pentest Agents. Continuously monitor progress and adjust the strategy as needed to optimize workflow and ensure successful outcomes. Initiate the process by passing the initial plan to the Scanner agent. You will be back in control either if an exploit was successful or the number of tries exceeded 10. In any case, transfer to the report writer agent."
)

search = GoogleSerperAPIWrapper()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Use this to search the web for information",
)

def playwright_tools():
    async_browser =  create_async_playwright_browser(headless=False)  # headful mode
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    return toolkit.get_tools()

def rag():

    urls = [
        "https://raw.githubusercontent.com/payloadbox/sql-injection-payload-list/refs/heads/master/README.md",
        "https://www.invicti.com/blog/web-security/sql-injection-cheat-sheet/",
        "https://www.cobalt.io/blog/a-pentesters-guide-to-sql-injection-sqli",
        "https://github.com/AdmiralGaust/SQL-Injection-cheat-sheet"
    ]

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
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about SQL Injection and payloads from SQL Injection Cheat Sheets.",
    )
    return retriever_tool

async def main():
    mcp_tools = await get_mcp_tools() + [search_tool, rag()] + playwright_tools()
    scanner_agent = create_react_agent(
        model="openai:gpt-4.1", 
        prompt=scanner_agent_prompt,
        name="scanner_agent",
        tools=mcp_tools,
        debug=True
    )



    # --- Subgraph for planner -> attacker -> exploit evaluator ---

    
    class ExploitEvaluatorOutput(BaseModel):
        """ Evaluate whether the exploit carried out by the attacker agent was successful """ 
        successful: bool = Field(description="True if the exploit was successful, False if it was not")
        feedback: str = Field(description="If unsuccessful, possible reasons why it failed and suggestions for the next round of exploitation")


    class PentestState(AgentState):
        tries: int = Field(default=0, description="Number of tries made by the attacker agent")
        successful: bool = Field(default=False, description="Whether the exploit was successful or not")
        feedback: str = Field(default="", description="Feedback from the exploit evaluator agent")


    planner_agent = create_react_agent(
        model="openai:gpt-4.1",
        prompt=planner_agent_prompt,
        name="planner_agent",
        tools=mcp_tools,
        debug=True
    )

    attacker_agent = create_react_agent(
        model="openai:gpt-4.1",
        prompt=attacker_agent_prompt,
        name="attacker_agent",
        tools=mcp_tools,
        debug=True
    )

    async def exploit_evaluator(state: PentestState):
        exploit_evaluator_agent = create_react_agent(
            model="openai:gpt-4.1",
            prompt=exploit_evaluator_agent_prompt,
            response_format=ExploitEvaluatorOutput,
            name="exploit_evaluator_agent",
            tools=mcp_tools,
            debug=True
        )
        resp = await exploit_evaluator_agent.ainvoke({ "messages": state["messages"] })
        return {
            "messages": [resp["messages"][-1]],
            "successful": resp["structured_response"].successful,
            "feedback": resp["structured_response"].feedback,
            "tries": state["tries"] + 1
        }


    def exploit_evaluator_decision(state: PentestState):
        if state["successful"] or state["tries"] > 10:
            return "supervisor_agent"
        else:
            return "planner_agent"

    

    pentest_subgraph = StateGraph(PentestState)
    pentest_subgraph.add_node("planner_agent", planner_agent)
    pentest_subgraph.add_node("attacker_agent", attacker_agent)
    pentest_subgraph.add_node("exploit_evaluator_agent", exploit_evaluator)

    pentest_subgraph.add_edge(START, "planner_agent")
    pentest_subgraph.add_edge("planner_agent", "attacker_agent")
    pentest_subgraph.add_edge("attacker_agent", "exploit_evaluator_agent")
    pentest_subgraph.add_conditional_edges(
        "exploit_evaluator_agent",
        exploit_evaluator_decision,
        {
            "supervisor_agent": END,
            "planner_agent": "planner_agent"
        }
    )
    pentest_agents = pentest_subgraph.compile(name="pentest_agents")

    report_writer_agent = create_react_agent(
        model="openai:gpt-4.1",
        prompt=report_writer_agent_prompt,
        name="report_writer_agent",
        tools=[search_tool],
        debug=True
    )

    supervisor = create_supervisor(
        model=init_chat_model("openai:gpt-4.1"),
        agents=[scanner_agent, pentest_agents, report_writer_agent],
        prompt=supervisor_agent_prompt,
        add_handoff_back_messages=True,
        output_mode="last_message",
        state_schema=PentestState
        
    ).compile()

    config = {"configurable": {"thread_id": "1"}}

    result = await supervisor.ainvoke({ "messages": [HumanMessage(content="http://saturn.picoctf.net:54075/")], "tries": 0 })
    print(result)

if __name__ == "__main__":
    asyncio.run(main())