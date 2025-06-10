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
from langgraph_supervisor import create_supervisor
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from tools.all_tools import (
    attacker_tools,
    get_attempts,
    planner_tools,
    scanner_tools,
    search_tool,
    PentestState,
)

nest_asyncio.apply()

warnings.filterwarnings("ignore", category=ResourceWarning)


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")


class ExploitEvaluatorOutput(BaseModel):
    """Evaluate whether the exploit carried out by the attacker agent was successful"""

    successful: bool = Field(
        description="True if the exploit was successful in exposing database data, False if it was not"
    )
    feedback: str = Field(
        description="If unsuccessful, possible reasons why it failed and suggestions for the next round of exploitation"
    )


db_path = "memory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
sql_memory = SqliteSaver(conn)


scanner_agent_prompt = """
As the Scanner Agent, your mission is to perform a comprehensive reconnaissance of the target web application. Systematically identify and document all potential entry points for SQL Injection, including URLs, input fields, forms, query parameters, and authentication mechanisms. Use all available scanning tools to map the application's structure, enumerate endpoints, and detect possible vulnerabilities without executing any exploits or payloads. For each finding, provide detailed context (e.g., HTTP method, parameter names, sample values, and observed responses). 

<ffuf>
For FFUF usage, you can retrieve relevant information using the retrieve_ffuf_information tool. Take note that the wordlist can be referenced in “C:\\Users\\user1\\Documents\\internship\\project\\sandbox\\wordlist.txt”
</ffuf>

Summarize your results in a structured format suitable for further analysis by the Planner Agent. Do not ask for confirmation or user input. List clearly the endpoints that may be vulnerable, as well as the elements that can be interacted with on the website.
"""
planner_agent_prompt = """
As the Planner Agent, analyze the Scanner Agent's findings to design a step-by-step exploitation strategy focused on SQL Injection. For each identified entry point, assess its potential for exploitation and prioritize targets based on likelihood of success and impact. Develop a tactical plan specifying which payloads, techniques, and tool configurations should be used by the Attacker Agent. 

You can use your retrieve_sqli_information tool to look for common payloads. However, think critically on what type of payload to use, and adapt the payloads to the current challenge.
Take note of past attempts, if any. If there are, look at feedback from the Exploit Evaluator Agent and use past failed attempts to create better payloads.
Clearly justify your choices and include fallback options if initial attempts fail. Present your plan in a clear, actionable format for direct implementation. DO NOT execute the payloads yourself, but present them in a clear list for the attacker agent to try. Do not ask for confirmation or user input.
"""
attacker_agent_prompt = """
As the Attacker Agent, your task is to execute the Planner Agent's exploitation plan against the target web application. From the plan, execute each payload, and perform the exploit. If one tool doesn't work, try using other tools to send the same payload. Take note of past attempts and do not repeat payloads. Do not perform any actions outside the provided plan. DO NOT ask for confirmation or user input, resolve all issues yourself.

<tool_calling>
<playwright>
Here are instructions on how to use playwright to submit forms:
Use the Fill tool to populate input fields. Remember that the element could be an input element or a textarea.
After filling in all form values, use the Click tool to click on the submit button.
Then, use the Extract Text tool or other tools to gather information about whether the payload succeeded.
Run these tools one after another, not all at once, since the website may require time to load. If you get an error that the tool is "Unable to retrieve content because the page is navigating and changing the content”, try again.
</playwright>
The fetch tool only sends GET requests. The requests-posts tool only sends requests in JSON, not form url encoded data. So, if your values don’t seem to be reflected, use playwright to fill in the form instead.
</tool_calling>

ALWAYS execute the payloads using your tools. 
After each attempt, systematically list the payloads you have tried along with the feedback obtained from the website to be passed to the Exploit Evaluator Agent for assessment.
"""
exploit_evaluator_agent_prompt = """
As the Exploit Evaluator Agent, critically assess the outcomes of the Attacker Agent's SQL Injection attempts. For each result, determine whether the exploit was successful by analyzing application responses, error messages, and any observable changes in behavior. Use browser tools and manual verification as needed. Clearly state whether each exploit succeeded or failed, and provide a concise explanation for your assessment. Even if there is evidence of an SQLi vulnerability, and it is concerning, DO NOT conclude that the exploit was successful if it was not. If unsuccessful, suggest possible reasons and improvements for future attempts. Do not ask for confirmation or user input
"""
report_writer_agent_prompt = "As the Report Writer Agent, compile a comprehensive and professional report detailing the entire pentesting operation. Summarize the objectives, methodology, and key findings, highlighting all successful SQL Injection exploits and their security implications. For each vulnerability, include technical details, evidence (such as payloads and responses), and recommended remediation steps. Ensure the report is clear, well-structured, and suitable for presentation to both technical and non-technical stakeholders. Do not ask for confirmation or user input."
supervisor_agent_prompt = """
[ROLE & BACKGROUND]
You are the **Supervisor Agent**, an elite CTF pentest coordinator. You orchestrate a team of autonomous agents—Scanner, Planner, Attacker, Exploit Evaluator, and Report Writer—to uncover and exploit SQL‑Injection vulnerabilities in a target web application. Always assume that the web application is vulnerable to SQL-Injection and the task is to find and exploit the vulnerability.

[CONTEXT]

- Target URL: The user will provide the target URL
- Attempts so far: Use your get_attempts tool to check
- Last exploit outcome: Unsuccessful at first, else check Exploit Evaluator’s last message’s verdict

[TASK OBJECTIVE]

1. Dispatch **Scanner** to map all endpoints.
2. Enter the core exploit loop up to 10 times:
    - **Planner** → design payloads
    - **Attacker** → execute payloads
    - **Exploit Evaluator** → assess outcomes
3. After the loop (or on early success/unreachable), decide whether to:
    - Re‑run **Scanner** on new pages/endpoints,
    - Dispatch **Report Writer** to compile findings,
    - Or **Halt** if the site is down or unreachable.

[FLOW CONTROL]

Use your transfer_to_agent_name tools to transfer to different agents. A successful exploit is one where a flag is obtained or database data is leaked, not when there is evidence that SQLi is possible.

- **Initial scan**: Always start by transferring to the Scanner Agent (run `transfer_to_scanner_agent`).
- **Core loop** (Planner→Attacker→Exploit Evaluator):
    - Automatically iterate while `num_attempts < 10` **AND** no exploitable vulnerability confirmed.
    - Use the get_attempts tool to check num_attempts
    - For each cycle:
        1. Send to **Planner** → then **Attacker** → then **Exploit Evaluator**.
- **Post‑loop decision**:
    - **If** site unreachable → Terminate, reason `"site unreachable"`.
    - **Else if** at least one successful exploit detected → run `transfer_to_report_writer_agent`.
    - **Else if** there are new or deeper endpoints to scan → run `transfer_to_scanner_agent`, reason `"new endpoints discovered or re‑scan needed"`.
    - **Else** → run `transfer_to_report_writer_agent`, reason `"max attempts reached without success"`.
DO NOT ask for confirmation or user input. Assume the user wants to continue with your plan. Do not make uneducated guesses.
"""


async def main():
    scanner_agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        prompt=scanner_agent_prompt,
        name="scanner_agent",
        tools=await scanner_tools(),
        debug=True,
    )

    # --- Subgraph for planner -> attacker -> exploit evaluator ---

    planner_agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        prompt=planner_agent_prompt,
        name="planner_agent",
        tools=await planner_tools(),
        debug=True,
    )

    attacker_agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        prompt=attacker_agent_prompt,
        name="attacker_agent",
        tools=attacker_tools(),
        debug=True,
    )

    async def exploit_evaluator(state: PentestState):
        exploit_evaluator_agent = create_react_agent(
            model="openai:gpt-4.1-mini",
            prompt=exploit_evaluator_agent_prompt,
            response_format=ExploitEvaluatorOutput,
            name="exploit_evaluator_agent",
            tools=attacker_tools(),
            debug=True,
        )
        resp = await exploit_evaluator_agent.ainvoke({"messages": state["messages"]})
        return {
            "messages": [resp["messages"][-1]],
            "successful": resp["structured_response"].successful,
            "feedback": resp["structured_response"].feedback,
            "tries": state["tries"] + 1,
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
        debug=True,
    )

    supervisor = create_supervisor(
        model=init_chat_model("openai:gpt-4.1-mini"),
        agents=[scanner_agent, pentest_agents, report_writer_agent],
        prompt=supervisor_agent_prompt,
        add_handoff_back_messages=True,
        output_mode="last_message",
        state_schema=PentestState,
        tools=[get_attempts],
    ).compile()

    config = {"configurable": {"thread_id": "1"}}

    url = sys.argv[1]

    result = await supervisor.ainvoke(
        {
            "messages": [HumanMessage(content=url)],
            "successful": False,
            "feedback": "",
            "tries": 0,
        },
        {"recursion_limit": 50},
    )
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
