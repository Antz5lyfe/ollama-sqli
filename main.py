import getpass
import os
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langgraph_supervisor import create_supervisor
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage
from mcp_client import get_mcp_tools

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


_set_if_undefined("OPENAI_API_KEY")

scanner_agent_prompt = (
            "As the Scanner agent, your role is to gather crucial information about the target web application," ""
            "identifying potential entry points for SQL Injection attacks. "
            "You utilize advanced scanning tools to map the application's structure, "
            "locate input fields, and pinpoint possible attack URLs."
            "You are provided with many tools, and it is up to you to decide which tools to use."
            "DO NOT perform exploits and execute payloads, just scan for vulnerabilities and input fields that could be used for SQL Injection. "
            "Your goal is to provide a comprehensive overview of the target's attack surface. "
            "Conduct a thorough reconnaissance of the target application, documenting all relevant details, "
            "including input fields, form parameters, and potential vulnerabilities. "
            "Compile your findings into a detailed report and pass this information to the "
            "Planner agent for the next steps in the exploitation process.")

planner_agent_prompt = (
    "As the Planner agent, you are responsible for devising the next steps in the exploitation process based on the Scanner agent's findings. Your strategic insights are crucial in determining the most effective approach to exploit the identified vulnerabilities. Your goal is to create a tactical plan for executing SQL Injection attacks. Analyze the data provided by the Scanner agent and develop a plan that outlines the specific payloads and methodologies to be used by the Attacker agent. Ensure that the plan is adaptable and can be revised based on feedback from the Exploit Evaluator. Create detailed and clear steps on how to exploit a promising vulnerability found. Pass the plan to the Attacker agent."
)

attacker_agent_prompt = (
    "As the Attacker agent, your mission is to execute the planned SQL Injection attacks on the target web application. You craft and deploy payloads to exploit vulnerabilities, testing the application's defenses and uncovering potential security breaches. Your goal is to effectively exploit the identified vulnerabilities. Implement the plan provided by the Planner agent, crafting precise payloads and executing them against the target application. Document the outcomes of each attack attempt and pass the results to the Exploit Evaluator agent for assessment."
)

exploit_evaluator_agent_prompt = (
    "As the Exploit Evaluator agent, you are to evaluate whether the exploit carried out by the Attacker agent worked. Use your browser tools to check whether the SQL Injection has worked."
)

report_writer_agent_prompt = (
    "As the Report Writer agent, your responsibility is to compile a comprehensive and detailed report of the pentesting operation, highlighting the successful exploits and providing insights into the vulnerabilities of the target application. Your goal is to produce a clear and informative report for stakeholders. Utilize the information provided by the Exploit Evaluator agent to document the entire process, emphasizing the successful SQL Injection exploits and their implications. Ensure the report is thorough, professional, and suitable for presentation to stakeholders."
)

supervisor_agent_prompt = (
    "As the Supervisor of the pentesting AI app, your mission is to orchestrate the entire red teaming process, ensuring a strategic and efficient approach to uncovering vulnerabilities in CTF-style web applications. You are responsible for defining the overall plan and strategy, coordinating with other agents, and ensuring that each step aligns with the ultimate goal of identifying and exploiting security weaknesses. Your goal is to establish a comprehensive and adaptable strategy for the pentesting operation. Begin by outlining the objectives and key phases of the red teaming exercise. Set clear guidelines and priorities for the Scanner, Planner, Attacker, Exploit Evaluator, and Report Writer agents. Continuously monitor progress and adjust the strategy as needed to optimize workflow and ensure successful outcomes. Initiate the process by passing the initial plan to the Scanner agent."
)

async def main():
    mcp_tools = await get_mcp_tools()

    scanner_agent = create_react_agent(
        model="openai:gpt-4.1", 
        prompt=scanner_agent_prompt,
        name="scanner_agent",
        tools=[mcp_tools]    
    )

    planner_agent = create_react_agent(
        model="openai:gpt-4.1",
        prompt=planner_agent_prompt,
        name="planner_agent"
    )

    attacker_agent = create_react_agent(
        model="openai:gpt-4.1",
        prompt=attacker_agent_prompt,
        name="attacker_agent",
        tools=[mcp_tools]
    )

    class ExploitEvaluatorOutput(BaseModel):
        """ Evaluate whether the exploit carried out by the attacker agent was successful """ 
        successful: bool = Field(description="True if the exploit was successful, False if it was not")


    exploit_evaluator_agent = create_react_agent(
        model="openai:gpt-4.1",
        prompt=exploit_evaluator_agent_prompt,
        response_format=ExploitEvaluatorOutput,
        name="exploit_evaluator_agent"
    )


    # --- Subgraph for planner -> attacker -> exploit evaluator ---

    def exploit_evaluator_decision(state: MessagesState):
        # Assumes the last message contains the ExploitEvaluatorOutput
        last_message = state["messages"][-1]
        # If using structured output, last_message.content is a dict
        if hasattr(last_message, "content") and isinstance(last_message.content, dict):
            if last_message.content.get("successful"):
                return "supervisor_agent"
            else:
                return "planner_agent"
        # fallback: treat as unsuccessful
        return "planner_agent"

    pentest_subgraph = StateGraph(MessagesState)
    pentest_subgraph.add_node("planner_agent", planner_agent)
    pentest_subgraph.add_node("attacker_agent", attacker_agent)
    pentest_subgraph.add_node("exploit_evaluator_agent", exploit_evaluator_agent)

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
    pentest_subgraph.compile()

    report_writer_agent = create_react_agent(
        model="openai:gpt-4.1",
        prompt=report_writer_agent_prompt,
        name="report_writer_agent"
    )

    supervisor = create_supervisor(
        model=init_chat_model("openai:gpt-4.1"),
        agents=[scanner_agent, pentest_subgraph, report_writer_agent],
        prompt=supervisor_agent_prompt,
        add_handoff_back_messages=True,
        output_mode="full_history",
    ).compile()


    result = supervisor.invoke({ "messages": [HumanMessage(content="http://saturn.picoctf.net:64144/")] })
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())