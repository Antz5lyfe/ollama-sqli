import sys
from typing import TypedDict, Union

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
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from pydantic import Field

# from langchain_community.chat_models import ChatOllama
from langchain_ollama.chat_models import ChatOllama


from tools.all_tools import (
    PentestState,
    attacker_tools,
    get_attempts,
    planner_tools,
    report_writer_tools,
    scanner_tools,
)

nest_asyncio.apply()

warnings.filterwarnings("ignore", category=ResourceWarning)


def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


# _set_if_undefined("OPENAI_API_KEY")


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


class PlannerOutput(TypedDict):
    """Final output of the Agent"""

    final_output: dict[str, Union[str, list]] = Field(
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

[CRAWLING FLOW]

1. **Start at the Target URL**
    - Load the page with Playwright.
    - Use `extract_html` or `get_elements` to find forms and input fields
    - Use `extract_hyperlinks` to extract links
2. **Filter & Follow Promising Links**
    - From the list of links, select those whose text or URL path suggests a data-entry form or authentication page.
    - Navigate to each selected link (repeat steps 1–2 on that page).
3. **Map Endpoints on Every Page**
    
    For each visited page:
    
    - Identify all endpoints (GET & POST) and any query or form inputs.
    - Locate every HTML form or input element.
    - Record status codes, response contents, and any parameter reflection or error messages.

[MORE TOOLS USAGE INFORMATION]

- **Playwright**:
    - `extract_hyperlinks` → collect links
    - `navigate_browser` → visit pages in the links
    - `get_elements` → find `<form>`, `<input>`, `<textarea>` OR `extract_html` to look through the HTML source code of the website
    - `fill_element` to fill in forms and `click_element` to click buttons to submit forms
- **ffuf**: fuzz directories/parameters with `/Users/javiertan/internship/agentic-sqli/sandbox/wordlist.txt`
- **fetch**: search for information. Can only only GET, cannot POST
- **requests:** note that the POST tool can only send data in JSON, and does not support form encoded data, and so may not work for sending form values.

[EXPECTED OUTPUT]
Once crawling is complete, return a list of all entry points discovered. For each, include:

- **Page URL**: URL of the page with the input fields/form
- **Endpoint**: full URL + HTTP method
- **Parameters**: names + example values
- **Reflection/Error**: yes/no; if yes, include full SQL command fragment
- **Forms/Inputs**: form action URL + field names/types
- **Goal**: e.g. “bypass login,” “leak database items”

Return only that list in a clear, structured format. Do not ask for user confirmation—crawl until you’ve exhaustively mapped all entry points.
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
- **Attempt History**:  {attempts}

[MEMORY SUMMARY]
Review the Attempt History array and summarize its key points in a few bullets, for example:

- Payload `' OR 1=1--` at `/login` reflected as expected but did not return a welcome page (no auth bypass). From examining the reflected command, I see that the command started with the password field instead of the username field, so I should inject in that field instead.
- Payload `' UNION SELECT null, table_name FROM information_schema.tables--` at `/login` produced a “column count mismatch” error. This could mean that I should create payloads with more nulls until I do not get an error.
- Etc.

[TASK OBJECTIVE]
For each potential SQLi entry point discovered:

**Phase 1: Failure Analysis (Prose)**  

1. **Think step by step** about each past attempt:
    - Extract any reflected SQL from `response_excerpt` and explain how the payload was interpolated.
    - Identify specific tweaks (comments, column counts, encoding, trying different field) needed.

**Phase 2: Plan Generation**

1. Determine current objectives. For example, this attempt could be to gather information that will be considered for future attempts (such as determining database type by using provider-specific queries).
2. Using your analysis, craft **3-4 payloads** per entry point. Here are some possible SQL injection types:
    - Simple comment based bypass (username: admin' --)
    - Simple **Boolean-based** tests
    - **Error-based** probing
    - Database-Type Discovery (e.g. version functions)
    - Schema Enumeration (information_schema or catalog tables)
    - **UNION-based** payloads to retrieve data from other tables within the database
    - Other types you think are relevant (stacked queries)
3. **Craft payloads**
    - Remember that you can use comments to invalidate the back part of the query.
    - You do not have to use all types of payloads
    - For each payload entry, ensure you include a `"payloads"` object mapping **every** input field name to its payload value.
    - The Attacker Agent cannot send POST requests, only navigate to a page and fill in a form. The endpoint should be the form page and the payload should only have the fields in the form
    
[OUTPUT FORMAT]

1. **Failure Analysis** (prose): a short paragraph summarizing your findings.
2. **Plan** (JSON array of objects):

```json
[
    {{
        "entry_point": "<URL>",
        "page_url": "<URL of the page with the form>",
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
    }},
    …
]
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
You are the **Attacker Agent**, an elite exploit developer specialized in SQL‑Injection execution. You take the Planner Agent's payload playbook and carry out each injection attempt against the target application, adapting tactics as needed.

[CURRENT CONTEXT]

- **Plans from Planner Agent**: {payloads}
- **Recommendation from Critic Agent**: {recommendation}

[TASK OBJECTIVE]
For each entry point:

1. Execute each **payload** from the Planner Agent and Critic Agent in order.
2. Use **Playwright** first, before trying other methods
3. **Capture Outcomes**
    - Record HTTP status code, any reflected input or error text, and a short excerpt of the page response.
    - Retry once on navigation errors before falling back.
4. **Document Every Attempt**
    - Prepare structured results for the Exploit Evaluator.

[TOOL GUIDANCE]

- **Playwright (Main tool)**
    1. **Load the target page containing the form**
        
        Use `navigate_browser` to navigate to `page_url`
        
    2. **Locate inputs**
        
        Use `get_elements` for input elements like `<input>`, `<textarea>`, `<button>`. Use the Scanner Agent's findings to verify input elements
        
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
    "entry_point":"<URL>",
    "page_url": "<URL of the page with the form>",
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
          "entry_point": "<URL>",
          "page_url": "<URL of the page with the form>",
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
3. The scanner needs to be called to scan a new page. If so, terminate the loop, as the scanner is separate from the loop. State that more scanning is needed in your reason.

Even if there is evidence of an SQLi vulnerability, and it is concerning, DO NOT terminate yet, until the vulnerability has been completely exploited. If termination is necessary, provide a reason. Do not ask for confirmation or user input. Do not issue recommendations.
"""

report_writer_agent_prompt = ChatPromptTemplate(
    [
        (
            "system",
            """
[ROLE & BACKGROUND]

You are the **Report Writer Agent**, a concise technical writer specializing in pentest documentation. Your goal is to transform the raw exploit attempts and final success into a clear, professional markdown report.

[CONTEXT]

- **Attempt history:** {attempts}
- **Successful payload**: {successful_payload}

[TASK OBJECTIVE]

Using the attempt history and the confirmed successful payload, produce a concise yet insightful report that includes:

1. **Executive Summary**
    - One-paragraph overview of objectives and outcome.
2. **Methodology**
    - Briefly describe each phase (Scanning → Planning → Attacking → Evaluation → Critique).
3. **Key Findings**
    - Bullet-list of tested entry points, observed failure modes, and the one that succeeded.
4. **Successful Exploit Details**
    - Show the final payload mapped to each field, explain why it worked.
5. **Security Implications & Recommendations**
    - Outline the vulnerability's impact and suggest remediation steps.
6. **Lessons Learned & Next Steps**
    - Note any patterns (e.g., WAF quirks, filtering) and propose further testing or defensive measures.

[OUTPUT FORMAT]

1. **Markdown Document**: Generate a single markdown file named `report.md` with appropriately leveled headings (`#`, `##`, ), code blocks for payload examples, and tables or lists where helpful.
2. **File Creation**: Use your file management tools (e.g. `write_file`) to write the markdown content to `report.md`.
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
- A successful exploit must extract a flag or database data, or bypass authentication—errors or evidence of injection alone don't count.

Proceed strategically and efficiently to maximize success in exploiting vulnerabilities.
""",
        ),
        ("placeholder", "{messages}"),
    ]
)


async def main():
    scanner_agent = create_react_agent(
        # model="openai:gpt-4.1-mini",
        # model=ChatOllama(model="mistral:7b-instruct"),
        model=ChatOllama(model="qwen3:14b"),
        prompt=scanner_agent_prompt,
        name="scanner_agent",
        tools=await scanner_tools(),
        state_schema=PentestState,
        debug=True,
    )

    # --- Subgraph for planner -> attacker -> exploit evaluator ---
    async def planner(state: PentestState):
        planner_agent = create_react_agent(
            # model="openai:o4-mini",
            # model=ChatOllama(model="mistral:7b-instruct"),
            model=ChatOllama(model="qwen3:14b"),
            prompt=planner_agent_prompt,
            name="planner_agent",
            tools=await planner_tools(),
            state_schema=PentestState,
            response_format=(
                """
    Copy the exact final JSON output. It should look like this:
    CRITICAL: The value for 'final_output' MUST be a JSON list (inside square brackets []), even if there is only one item.
    ```json
    {
        final_output: [
            {
                "entry_point": "...",
                "page_url": "...",
                "payload_sequence": [
                    {
                        "type": "...",
                        "payloads": {
                            "<field_name_1>": "...",
                            "<field_name_2>": "...",
                            …           : …
                        },
                        "reason": "..."
                    },
                    …
                ],
                "justification": "..."
            }
        ]
    }
    ```
    """,
                PlannerOutput,
            ),
            debug=True,
        )
        '''
        resp = await planner_agent.ainvoke(state)
        if "final_output" not in resp["structured_response"] or not isinstance(
            resp["structured_response"]["final_output"], list
        ):
            raise ValueError("Planner agent did not return payloads")
        return {
            "messages": [resp["messages"][-1]],
            "payloads": resp["structured_response"]["final_output"],
        }
        '''
        
        resp = await planner_agent.ainvoke(state)
        # Check if the response is valid
        if "final_output" not in resp["structured_response"]:
            raise ValueError("Planner agent did not return 'final_output'")

        final_output = resp["structured_response"]["final_output"]

        # if the model returned a dict instead of a list
        if isinstance(final_output, dict):
            final_output = [final_output]

        # Ensure it's a list
        if not isinstance(final_output, list):
            raise ValueError("Planner agent did not return payloads in a valid list format")

        return {
            "messages": [resp["messages"][-1]],
            "payloads": final_output, # Use the corrected variable
        }

    async def attacker(state: PentestState):
        attacker_agent = create_react_agent(
            # model="openai:gpt-4.1-mini",
            # model=ChatOllama(model="mistral:7b-instruct"),
            model=ChatOllama(model="qwen3:14b"),
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
            "entry_point":"<URL>",
            "page_url": "<URL of the page with the form>",
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
            # model="openai:gpt-4.1-mini",
            # model=ChatOllama(model="mistral:7b-instruct"),
            model=ChatOllama(model="qwen3:14b"),
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
                "entry_point": "<URL>",
                "page_url": "<URL of the page with the form>",
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
                    analysis_entry["page_url"] == attempt_entry["page_url"]
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
            # model="openai:gpt-4.1-mini",
            # model=ChatOllama(model="mistral:7b-instruct"),
            model=ChatOllama(model="qwen3:14b"),
            prompt=exploit_evaluator_agent_prompt,
            response_format=(exploit_evaluator_agent_prompt, ExploitEvaluatorOutput),
            name="exploit_evaluator_agent",
            tools=[],
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
    pentest_subgraph.add_node("planner_agent", planner)
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
        # model="openai:gpt-4.1-mini",
        # model=ChatOllama(model="mistral:7b-instruct"),
        model=ChatOllama(model="qwen3:14b"),
        prompt=report_writer_agent_prompt,
        name="report_writer_agent",
        tools=report_writer_tools(),
        state_schema=PentestState,
        debug=True,
    )

    supervisor = create_supervisor(
        # model=init_chat_model("openai:gpt-4.1-mini"),
        # model=ChatOllama(model="mistral:7b-instruct"),
        model=ChatOllama(model="qwen3:14b"),
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
            "successful_payload": None,
            "payloads": [],
            "structured_response": {},
        },
        {"recursion_limit": 100},
    )


if __name__ == "__main__":
    asyncio.run(main())
