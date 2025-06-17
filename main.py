import sys
from typing import TypedDict, Union

from langchain_openai import ChatOpenAI

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


class ExploitEvaluatorOutput(TypedDict):
    """Evaluate whether the exploit carried out by the attacker agent was successful"""

    should_terminate: bool = Field(
        description="True if the pentest loop should terminate"
    )
    reason: str = Field(description="Reason for verdict")
    successful_payload: Union[None, dict[str, str]] = Field(
        description="""
If the loop should terminate and the exploit was successful, this will contain the payload that were successful for each field.
It should look like this:
```json
"payloads": {
    "<field_name_1>": "<payload for field 1>",
    "<field_name_2>": "<payload for field 2>",
    …           : …
}
```
Else, this field should be empty/null.
"""
    )


class AttackerOutput(TypedDict):
    """Final output of the Agent"""

    final_output: list[dict[str, Union[str, dict]]] = Field(
        description="The exact final json output of the Agent"
    )


class CriticOutput(TypedDict):
    """Final JSON output of the Agent"""

    final_output: dict[str, Union[list[dict], dict]] = Field(
        description="The exact final json output of the Agent"
    )


db_path = "memory.db"
conn = sqlite3.connect(db_path, check_same_thread=False)
sql_memory = SqliteSaver(conn)


scanner_agent_prompt = ChatPromptTemplate(
    [
        (
            "system",
            """
[ROLE & BACKGROUND]
You are the **Scanner Agent**, a seasoned reconnaissance specialist tasked with non-intrusive mapping of a web application to uncover every potential SQL-Injection entry point.

[CONTEXT]

- Target URL: {url}

[TASK OBJECTIVE]

1. Conduct a **comprehensive, non-intrusive** scan of the target:
    - Enumerate directories, parameters, and paths.
    - Identify all endpoints (GET & POST) and any query or form inputs.
    - Record status codes, response contents, and any parameter reflection or error messages.
    - Locate every HTML form or input element that could accept user data.
2. Attempt to conduct error based SQL Injection on forms to test if they are vulnerable.
    - If SQL is reflected, state the full SQL command.

[MORE TOOLS USAGE INFORMATION]

- **ffuf**: use the wordlist at C:\\Users\\user1\\Documents\\internship\\project\\sandbox\\wordlist.txt to fuzz directories, parameters, and paths. use retrieve_ffuf_information to get usage information
- **fetch**: search for information and extract HTML. cannot POST, can only GET
- **requests:** probe each discovered endpoint for parameter reflection, status codes, and basic error messages. you can also inspect the raw HTML source of the site for information. note that the POST tool can only send data in JSON, and does not support form encoded data, and so may not work for sending form values.
- **Playwright**: look for input fields and other elements that can be exploited. use the fill_tool to fill in form values and submit. extract_html can be used to look through the HTML source code of the website

[EXPECTED OUTPUT]
At the end of your scan, summarize your findings as a list of entries. For each entry include:

- **Endpoint**: full URL and HTTP method
- **Parameters**: names and example values
- **Reflection/Error**: whether input is reflected, and any error text. if input is reflected, state the full sql command
- **Forms/Inputs**: form action URL, field names/types
- **Goal**: what the goal is. For example, if it is a login form, the goal would be to bypass authentication and log in. If database items are shown, and there is an input field, the goal could be to leak important database items.

Return only that list in a clear, structured format. Do not ask for user confirmation—proceed until you’ve exhaustively mapped all entry points.
""",
        ),
        ("placeholder", "{messages}"),
    ]
)
planner_agent_prompt = ChatPromptTemplate(
    [
        (
            "system",
            """
[ROLE & BACKGROUND]
You are the **Planner Agent**, a professional penetration tester and attack strategist with deep expertise in SQL‑Injection methodologies. Your job is to transform raw scan data into a precise, prioritized exploitation playbook.

[CURRENT CONTEXT]

- **Scanner findings**: Provided by the Scanner Agent in previous messages
- **Attempt history**:  {attempts}

[TASK OBJECTIVE]
For each potential SQLi entry point discovered:

1. **Review Past Attempts**
    - If past attempts exist, analyze failures and refine payload choices. Look at the response excerpt. If the SQL injection is reflected, analyse the command and think step by step about why the payload failed. DO NOT repeat past payloads.
2. **Analyze feasibility**
    - Use a **step-by-step reasoning** approach and prioritise possible injection types:
        - Simple **Boolean-based** tests
        - **Error-based** probing
        - Database-Type Discover (e.g. trying different version commands)
        - Schema Enumeration (Based on database type discovered)
        - **UNION-based** payloads to retrieve data from other *tables* within the database
        - Other types you think are relevant
3. Prioritise and order payloads
    - Determine current objectives. For example, this attempt could be to gather information that will be considered for future attempts (such as determining database type by using provider-specific queries). You do not have to use all different types of payloads. You can use as many payloads as you like. By default, there should be 2-3 payloads per attempt.
4. **Craft payloads**
    - Make sure to consider how many fields the form has, and ensure that you have provided a payload for every field.
    - Remember that you can use comments to invalidate the back part of the query.

[OUTPUT FORMAT]
Explain what the challenge is, and why previous attempts failed.
Then, return a list containing:

```json
{{
  "entry_point": "<URL & method>",
  "payload_sequence": [
    {{
      "type": "<boolean|union|…>",
      "payloads": {{
        "<field_name_1>": "<payload for field 1>",
        "<field_name_2>": "<payload for field 2>",
        …           : …
      }},
      "reason": "<rationale>"
    }},
    …
  ],
  "justification": "<brief summary of approach>"
}}
```

**Important:** Each `payload_sequence` entry must include a `payloads` object that maps **every** input field name (as discovered by the Scanner Agent for this entry point) to its corresponding payload string. Keys in `payloads` must exactly match the field names.
""",
        ),
        ("placeholder", "{messages}"),
    ]
)
attacker_agent_prompt = ChatPromptTemplate(
    [
        (
            "system",
            """
[ROLE & BACKGROUND]
You are the **Attacker Agent**, an elite exploit developer specialized in SQL‑Injection execution. You take the Planner Agent’s payload playbook and carry out each injection attempt against the target application, adapting tactics as needed.

[CURRENT CONTEXT]

- **Plans**: Payload Sequences provided by the Planner Agent

[TASK OBJECTIVE]
For each entry point:

1. Execute each **payload** in order.
2. Use **Playwright** first, before trying other methods
3. **Capture Outcomes**
    - Record HTTP status code, any reflected input or error text, and a short excerpt of the page response.
    - Retry once on navigation errors before falling back.
4. **Document Every Attempt**
    - Prepare structured results for the Exploit Evaluator.

[TOOL GUIDANCE]

- **Playwright (Main tool)**
    1. **Load the target page containing the form**
        
        Use `navigate_browser`
        
    2. **Locate inputs**
        
        Use `get_elements` for input elements like `<input>`, `<textarea>`, `<button>`. Use the Scanner Agent’s findings to verify input elements
        
    3. Populate each field.
        
        Use `fill_element`. Call the tools sequentially instead of simultaneously.
        
    4. Submit the form
        
        Find a way to submit the payload. For example, use `click_element` to click the submit button if there is one, or 
        
    5. Wait for navigation
    6. Capture page content and feedback or error messages for analysis.
        
        Use `extract_html` and analyse results. If you get an error that the tool is "Unable to retrieve content because the page is navigating and changing the content”, try again, because the page is navigating
        
    - Run these playwright tools one after another, not all at once, since the website may require time to load.
- **Requests**
    - Build GET/POST (JSON only) with the payload inserted.
    - If POST is not working, try using other tools

[OUTPUT FORMAT]
Return an array of objects, for each payload tested:

```json
[
  {{
    "entry_point":"<URL & method>",
    "payloads": {{
      "<field_name_1>": "<payload for field 1>",
      "<field_name_2>": "<payload for field 2>",
      …           : …
    }},
    "response_excerpt":"<excerpt of page response. only include relevant parts>",
    "notes":"<excerpt of page response. only include relevant parts. if sql injection is reflected, display that.>"
  }},
  …
]
```

Proceed through the plan methodically, do not ask for human input, and exhaustively record each payload attempt. DO NOT output and terminate before you have tested ALL payloads provided by the Planner Agent.
IMPORTANT: DO NOT hallucinate executing the payloads when you did not. Make sure to use your tools to execute each payload first.
""",
        ),
        ("placeholder", "{messages}"),
    ]
)
critic_agent_prompt = ChatPromptTemplate(
    [
        (
            "system",
            """
[ROLE & BACKGROUND]

You are the **Critic Agent**, an expert SQL-Injection analyst. You diagnose why each exploit payload failed and propose an improved payload likely to succeed.

[INPUT]

JSON array `attempts`:
{attempts}

[TASK]

1. **Reason in Prose**
    - **For each attempt**:
        - **Extract Reflection**: If `response_excerpt` shows a reflected SQL fragment, describe how the payload was interpolated and capture the full SQL statement.
        - **Analyze**
            - Explain how the payload was interpolated (using `reflection` if available) and why it failed. Use your knowledge of SQL syntax. Always assume that the payload failed and the goal was not accomplished.
    - **Aggregate & Recommend**
        - Summarize common patterns across all attempts and infer any database‐specific behaviors or filters.
        - Provide one **improved payload** (or, if the form has multiple fields, a `payloads` mapping of field names to payload strings) that addresses the diagnosed issues.
2. Generate a structured JSON output
After your reasoning, output **only** the following JSON object (no extra text):
    
    ```json
    {{
      "analysis": [
        {{
          "entry_point": "<URL & method>",
          "payloads": {{
            "<field_name_1>": "<payload for field 1>",
            "<field_name_2>": "<payload for field 2>",
            …           : …
          }},
          "reflection": "<full SQL or null>",
          "analysis": "<short explanation>"
        }},
        …
      ],
      "recommendation": {{
        "payloads": {{
          "<field1>": "<payload1>",
          "<field2>": "<payload2>",
          …
        }},
        "reason": "<why this payload should succeed>"
      }}
    }}
    ```
    

[IMPORTANT]

- First, write your full diagnostic reasoning in prose.
- Then, on a new line, output the valid JSON object exactly as specified above.
- Do not include any other text after the JSON. 
        """,
        ),
        ("placeholder", "{messages}"),
    ]
)
exploit_evaluator_agent_prompt = """
[ROLE & BACKGROUND]

You are the **Exploit Evaluator Agent**, an expert in assessing SQL-Injection exploit attempts. Your sole purpose is to decide whether the pentest loop should terminate, request a re-scan, or continue.

[TASK OBJECTIVE]  
Analyze all attempts and decide if the loop should terminate. Consider three things:

1. if the CTF flag retrieved, all important database items leaked or authentication was bypassed using SQL injection
2. The website is down
3. The scanner needs to be called to scan a new page

Even if there is evidence of an SQLi vulnerability, and it is concerning, DO NOT terminate yet, until the vulnerability has been completely exploited. If termination is necessary, provide a reason. Do not ask for confirmation or user input. Do not issue recommendations.
"""

report_writer_agent_prompt = ChatPromptTemplate(
    [
        (
            "system",
            """
As the Report Writer Agent, compile a comprehensive and professional report detailing the entire pentesting operation. Summarize the objectives, methodology, and key findings, highlighting all successful SQL Injection exploits and their security implications. For each vulnerability, include technical details, evidence (such as payloads and responses), and recommended remediation steps. Ensure the report is clear, well-structured, and suitable for presentation to both technical and non-technical stakeholders. Do not ask for confirmation or user input.
Here is the successful payload:
{successful_payload}
""",
        ),
        ("placeholder", "{messages}"),
    ]
)
supervisor_agent_prompt = ChatPromptTemplate(
    [
        (
            "system",
            """
[ROLE & BACKGROUND]
You are the **Supervisor Agent**, an experienced pentesting coordinator specializing in SQL Injection vulnerabilities for Capture-The-Flag (CTF) web exploitation challenges. You strategically coordinate a team of autonomous agents—Scanner, Planner, Attacker, Exploit Evaluator, and Report Writer—to uncover and exploit SQL‑Injection vulnerabilities in a target web application. 

[CONTEXT]

- Target URL: {url}
- Attempts so far: Use your get_attempts tool to check
- Max attempts: 10

[TASK OBJECTIVE]

1. **Initial Scan**
    - Immediately dispatch Scanner Agent.
2. **Post-Pentest Agents Loop Decision**
Based on the last exploit outcome and attempts count, choose exactly one action:
    - `"scanner_agent"` if new endpoints or major content changes detected
    - `"report_writer_agent"` if a successful exploit occurred or attempts == 10 or site is unreachable

[FLOW CONTROL]

Use your `transfer_to_agent_name` tools to direct the workflow strategically.

[IMPORTANT INSTRUCTIONS]

- **DO NOT** request user confirmation; assume continuous operation.
- **ALWAYS ASSUME** the web application is vulnerable to SQL Injection and your primary objective is to exploit it successfully.
- A successful exploit must extract a flag or database data, or bypass authentication—errors or evidence of injection alone don’t count.

Proceed strategically and efficiently to maximize success in exploiting vulnerabilities.
""",
        ),
        ("placeholder", "{messages}"),
    ]
)


async def main():
    scanner_agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        prompt=scanner_agent_prompt,
        name="scanner_agent",
        tools=await scanner_tools(),
        state_schema=PentestState,
        debug=True,
    )

    # --- Subgraph for planner -> attacker -> exploit evaluator ---

    planner_agent = create_react_agent(
        model="openai:o4-mini",
        prompt=planner_agent_prompt,
        name="planner_agent",
        tools=await planner_tools(),
        state_schema=PentestState,
        debug=True,
    )

    async def attacker(state: PentestState):
        attacker_agent = create_react_agent(
            model="openai:gpt-4.1-mini",
            prompt=attacker_agent_prompt,
            name="attacker_agent",
            tools=attacker_tools(),
            state_schema=PentestState,
            response_format=(
                """
Copy the exact final JSON output. It should look like this:
```json
{
    final_output: [
        {
            "entry_point":"<URL & method>",
            "payloads": {
                "<field_name_1>": "<payload for field 1>",
                "<field_name_2>": "<payload for field 2>",
                …           : …
            },
            "response_excerpt":"<excerpt of page response. only include relevant parts. if sql injection is reflected, display that.>",
            "notes":"Observations and evaluations"
        },
        …
    ]
}
```
""",
                AttackerOutput,
            ),
            debug=True,
        )
        resp = await attacker_agent.ainvoke(state)
        if "final_output" not in resp["structured_response"] or not isinstance(
            resp["structured_response"]["final_output"], list
        ):
            raise ValueError("Attacker agent did not return any attempts")
        # obj = resp["structured_response"]["final_output"]
        # new_dict = [
        #     {k: obj[v][k] for k in obj[v].keys() - {"response_excerpt"}} for v in obj
        # ]
        return {
            "messages": [resp["messages"][-1]],
            "attempts": state["attempts"] + resp["structured_response"]["final_output"],
        }

    async def critic(state: PentestState):
        critic_agent = create_react_agent(
            model="openai:gpt-4.1-mini",
            prompt=critic_agent_prompt,
            name="critic_agent",
            tools=await planner_tools(),
            state_schema=PentestState,
            response_format=(
                """
Copy the exact final JSON output. It should look like this:
```json
{
    "final_output": {
        "analysis": [
            {
                "entry_point": "<URL & method>",
                "payloads": {
                    "<field_name_1>": "<payload for field 1>",
                    "<field_name_2>": "<payload for field 2>",
                    …           : …
                },
                "reflection": "<full SQL or null>",
                "analysis": "<short explanation>"
            },
            …
        ],
        "recommendation": {
        "payloads": {
            "<field1>": "<payload1>",
            "<field2>": "<payload2>",
            …
        },
        "reason": "<why this payload should succeed>"
        }
    }
}
```         
            """,
                CriticOutput,
            ),
            debug=True,
        )
        resp = await critic_agent.ainvoke(state)
        if "final_output" not in resp["structured_response"] or not isinstance(
            resp["structured_response"]["final_output"], dict
        ):
            raise ValueError("Critic agent did not return the final output")
        c = state["attempts"].copy()
        for analysis_entry in resp["structured_response"]["final_output"]["analysis"]:
            for attempt_entry in c:
                if (
                    analysis_entry["entry_point"] == attempt_entry["entry_point"]
                    and analysis_entry["payloads"] == attempt_entry["payloads"]
                ):
                    attempt_entry.update(analysis_entry)
        return {
            "messages": [resp["messages"][-1]],
            "attempts": c,
            "recommendation": resp["structured_response"]["final_output"][
                "recommendation"
            ],
        }

    async def exploit_evaluator(state: PentestState):
        exploit_evaluator_agent = create_react_agent(
            model="openai:gpt-4.1-mini",
            prompt=exploit_evaluator_agent_prompt,
            response_format=(exploit_evaluator_agent_prompt, ExploitEvaluatorOutput),
            name="exploit_evaluator_agent",
            tools=attacker_tools(),
            state_schema=PentestState,
            debug=True,
        )
        resp = await exploit_evaluator_agent.ainvoke(state)
        if "reason" not in resp["structured_response"]:
            raise ValueError(
                "Exploit Evaluator agent did not provide a reason for termination"
            )
        if "should_terminate" not in resp["structured_response"]:
            raise ValueError(
                "Exploit Evaluator agent did not indicate whether to terminate or not"
            )

        return {
            "messages": [resp["messages"][-1]],
            "should_terminate": resp["structured_response"]["should_terminate"],
            "reason": resp["structured_response"]["reason"],
            "tries": state["tries"] + 1,
            "attempts": []
            if resp["structured_response"]["should_terminate"]
            else state["attempts"],
            "recommendation": ""
            if resp["structured_response"]["should_terminate"]
            else state["recommendation"],
            "successful_payload": resp["structured_response"].get(
                "successful_payload", {}
            ),
        }

    def exploit_evaluator_decision(state: PentestState):
        if state["should_terminate"] or state["tries"] > 10:
            return "supervisor_agent"
        else:
            return "critic_agent"

    pentest_subgraph = StateGraph(PentestState)
    pentest_subgraph.add_node("planner_agent", planner_agent)
    pentest_subgraph.add_node("attacker_agent", attacker)
    pentest_subgraph.add_node("critic_agent", critic)
    pentest_subgraph.add_node("exploit_evaluator_agent", exploit_evaluator)

    pentest_subgraph.add_edge(START, "planner_agent")
    pentest_subgraph.add_edge("planner_agent", "attacker_agent")
    pentest_subgraph.add_edge("attacker_agent", "exploit_evaluator_agent")
    pentest_subgraph.add_conditional_edges(
        "exploit_evaluator_agent",
        exploit_evaluator_decision,
        {"supervisor_agent": END, "critic_agent": "critic_agent"},
    )
    pentest_subgraph.add_edge("critic_agent", "planner_agent")
    pentest_agents = pentest_subgraph.compile(name="pentest_agents")

    report_writer_agent = create_react_agent(
        model="openai:gpt-4.1-mini",
        prompt=report_writer_agent_prompt,
        name="report_writer_agent",
        tools=[search_tool],
        state_schema=PentestState,
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

    url = sys.argv[1]

    result = await supervisor.ainvoke(
        {
            "messages": [HumanMessage(content=url)],
            "tries": 0,
            "should_terminate": False,
            "reason": "",
            "url": url,
            "attempts": [],
            "recommendation": {},
            "structured_response": {},
        },
        {"recursion_limit": 100},
    )


if __name__ == "__main__":
    asyncio.run(main())
