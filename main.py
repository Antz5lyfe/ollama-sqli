import sys

if len(sys.argv) < 2:
    print("Usage: python main.py <url>")
    print("Please provide the target URL as the first argument.")
    sys.exit(1)

import asyncio
import getpass
import os
import sqlite3
import warnings

import nest_asyncio
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph_supervisor import create_supervisor
from pydantic import BaseModel, Field

from tools.all_tools import (all_tools, attacker_tools, planner_tools,
                             scanner_tools, search_tool)

nest_asyncio.apply()

warnings.filterwarnings("ignore", category=ResourceWarning)


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")

db_path = "memory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
sql_memory = SqliteSaver(conn)


scanner_agent_prompt = ("""
As the Scanner Agent, your mission is to perform a comprehensive reconnaissance of the target web application. Systematically identify and document all potential entry points for SQL Injection, including URLs, input fields, forms, query parameters, and authentication mechanisms. Use all available scanning tools to map the application's structure, enumerate endpoints, and detect possible vulnerabilities without executing any exploits or payloads. For each finding, provide detailed context (e.g., HTTP method, parameter names, sample values, and observed responses). 

<ffuf>
For FFUF usage, you can retrieve relevant information using the retrieve_ffuf_information tool. Take note that the wordlist can be referenced in “/Users/javiertan/internship/agentic-sqli/sandbox/wordlist.txt”
</ffuf>

Summarize your results in a structured format suitable for further analysis by the Planner Agent. Do not ask for confirmation or user input. List clearly the endpoints that may be vulnerable, as well as the elements that can be interacted with on the website.
""")
planner_agent_prompt = ("""
As the Planner Agent, analyze the Scanner Agent's findings to design a step-by-step exploitation strategy focused on SQL Injection. For each identified entry point, assess its potential for exploitation and prioritize targets based on likelihood of success and impact. Develop a tactical plan specifying which payloads, techniques, and tool configurations should be used by the Attacker Agent. You are provided with many tools, and make sure to use relevant tools. 
Take note of past attempts, if any. If there are, look at feedback from the Exploit Evaluator Agent and use past failed attempts to create better payloads.
Clearly justify your choices and include fallback options if initial attempts fail. Present your plan in a clear, actionable format for direct implementation. DO NOT execute the payloads yourself, but present them in a clear list for the attacker agent to try. Do not ask for confirmation or user input.
""")
attacker_agent_prompt = ("""
As the Attacker Agent, your task is to execute the Planner Agent's exploitation plan against the target web application. From the plan, execute each payload, and perform the exploit. If one tool doesn't work, try using other tools to send the same payload. Take note of past attempts and do not repeat payloads. Do not perform any actions outside the provided plan. DO NOT ask for confirmation or user input, resolve all issues yourself.
<playwright>
Here are instructions on how to use playwright to submit forms:
Use the Fill tool to populate input fields. Remember that the element could be an input element or a textarea.
After filling in all form values, use the Click tool to click on the submit button.
Then, use the Extract Text tool or other tools to gather information about whether the payload succeeded.
Run these tools one after another, not all at once, since the website may require time to load. If you get an error that the tool is "Unable to retrieve content because the page is navigating and changing the content”, try again.
</playwright>
The fetch tool only sends GET requests. The requests-posts tool only sends requests in JSON, not form url encoded data. So, if your values don’t seem to be reflected, use playwright to fill in the form instead.
ALWAYS execute the payload using your tools before returning a text response. After each attempt, summarize the results and pass them to the Exploit Evaluator Agent for assessment.
"""
)
exploit_evaluator_agent_prompt = ("""
As the Exploit Evaluator Agent, critically assess the outcomes of the Attacker Agent's SQL Injection attempts. For each result, determine whether the exploit was successful by analyzing application responses, error messages, and any observable changes in behavior. Use browser tools and manual verification as needed. Clearly state whether each exploit succeeded or failed, and provide a concise explanation for your assessment. Even if there is evidence of an SQLi vulnerability, and it is concerning, DO NOT conclude that the exploit was successful if it was not. If unsuccessful, suggest possible reasons and improvements for future attempts. Do not ask for confirmation or user input
""")
report_writer_agent_prompt = ("As the Report Writer Agent, compile a comprehensive and professional report detailing the entire pentesting operation. Summarize the objectives, methodology, and key findings, highlighting all successful SQL Injection exploits and their security implications. For each vulnerability, include technical details, evidence (such as payloads and responses), and recommended remediation steps. Ensure the report is clear, well-structured, and suitable for presentation to both technical and non-technical stakeholders. Do not ask for confirmation or user input.")
supervisor_agent_prompt = ("""
As the Supervisor of the capture-the-flag competition web exploitation challenge solver application, your mission is to orchestrate the entire SQL Injection red teaming process ensuring a strategic and efficient approach to uncovering vulnerabilities in CTF-style web applications. You are responsible for defining the overall plan and strategy, coordinating with other agents, and ensuring that each step aligns with the ultimate goal of identifying and exploiting security weaknesses. Your goal is to establish a comprehensive and adaptable strategy for the pentesting operation and solve the CTF web exploitation challenge.

Set clear guidelines and priorities for the Scanner, Planner, Attacker, Exploit Evaluator, and Report Writer agents. The Planner, Attacker and Exploit Evaluator agents are under Pentest Agents. Use your tools to transfer to other agents. Initiate the process by passing the initial plan to the Scanner agent. 

After being passed control by the Pentest Agents, evaluate if there is a need to run the Scanner agent again, for example, if you think there is a lack of information and the Scanner Agent could gather that information.
Else, if the website is down or is unable to be reached, terminate.
Else, pass control back to the Pentest Agents. After 10 attempts, or if an exploit attempt was successful, transfer to the Report Writer Agent to write a report about the operation.
""")

async def main():
    tools = await all_tools()
    scanner_agent = create_react_agent(
        model="openai:gpt-4.1-mini", 
        prompt=scanner_agent_prompt,
        name="scanner_agent",
        tools=await scanner_tools(),
        debug=True
    )
    
    # --- Subgraph for planner -> attacker -> exploit evaluator ---
    
    class ExploitEvaluatorOutput(BaseModel):
        """ Evaluate whether the exploit carried out by the attacker agent was successful """ 
        successful: bool = Field(description="True if the exploit was successful in exposing database data, False if it was not")
        feedback: str = Field(description="If unsuccessful, possible reasons why it failed and suggestions for the next round of exploitation")


    class PentestState(AgentState):
        tries: int
        successful: bool
        feedback: str


    planner_agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        prompt=planner_agent_prompt,
        name="planner_agent",
        tools=planner_tools(),
        debug=True
    )

    attacker_agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        prompt=attacker_agent_prompt,
        name="attacker_agent",
        tools=attacker_tools(),
        debug=True
    )

    async def exploit_evaluator(state: PentestState):
        exploit_evaluator_agent = create_react_agent(
            model="openai:gpt-4.1-mini",
            prompt=exploit_evaluator_agent_prompt,
            response_format=ExploitEvaluatorOutput,
            name="exploit_evaluator_agent",
            tools=attacker_tools(),
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
    pentest_subgraph.add_edge("exploit_evaluator_agent", END)
    pentest_agents = pentest_subgraph.compile(name="pentest_agents")

    report_writer_agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        prompt=report_writer_agent_prompt,
        name="report_writer_agent",
        tools=[search_tool],
        debug=True
    )

    supervisor = create_supervisor(
        model=init_chat_model("openai:gpt-4.1-mini"),
        agents=[scanner_agent, pentest_agents, report_writer_agent],
        prompt=supervisor_agent_prompt,
        add_handoff_back_messages=True,
        output_mode="last_message",
        state_schema=PentestState
        
    ).compile()

    config = {"configurable": {"thread_id": "1"}}

    url = sys.argv[1]

    result = await supervisor.ainvoke({ "messages": [HumanMessage(content=url)], "tries": 0 }, {"recursion_limit": 50})
    print(result)

if __name__ == "__main__":
    asyncio.run(main())