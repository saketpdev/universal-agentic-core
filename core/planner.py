import json
import logging
from typing import List
from models.state import SubTask, ExecutionPlan
from skills.orchestrator import load_registry
from core.llm import call_llm  # Assuming your standard LLM call function

logger = logging.getLogger("AgenticCore.Planner")

def generate_execution_plan(user_prompt: str) -> List[SubTask]:
    """Reads the user prompt and generates a strict DAG execution array."""
    logger.info("Planner: Deconstructing user prompt into execution graph.")

    # 1. Dynamically load available agents from the registry
    registry = load_registry()
    agent_roster = []

    for agent_name, config in registry.items():
        # We don't want the planner routing to itself or the old supervisor
        if agent_name in ["supervisor", "planner"]: 
            continue

        # Use the trigger keywords as a makeshift description for the Planner
        keywords = ", ".join(config.get("trigger_keywords", []))
        agent_roster.append(f"- **{agent_name}**: Best for handling: {keywords}")

    roster_string = "\n".join(agent_roster)

    # 2. Build the strict System Prompt
    system_prompt = f"""# ROLE: Enterprise DAG Planner
You are the Orchestrator for a multi-agent backend. The user will give you a complex request. Your job is to decompose it into a sequential Execution Plan.

## AVAILABLE WORKER AGENTS:
{roster_string}

## RULES:
1. You must output a JSON object containing a `tasks` array.
2. Every task must specify a valid `agent_target` from the available roster ONLY. Do not hallucinate agents.
3. Every task must have a specific, narrow `instruction` for that agent.
4. Sequence matters. If Task 2 depends on data from Task 1, order them correctly.
5. If the user asks for something outside our capabilities, create a single task for 'customer_support' to handle the rejection.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Deconstruct this request: {user_prompt}"}
    ]

    # 3. Call the LLM (Force JSON mode / Structured Output)
    try:
        response = call_llm(
            messages=messages, 
            response_schema=ExecutionPlan.model_json_schema(),
            tier="planner"  # Explicitly request the high-IQ model
        )
        # Parse the raw JSON string back into our Pydantic model
        plan_data = json.loads(response.content)
        validated_plan = ExecutionPlan(**plan_data)

        logger.info(f"Planner successfully generated {len(validated_plan.tasks)} sub-tasks.")
        return validated_plan.tasks

    except Exception as e:
        logger.error(f"Planner failed to generate a valid DAG: {str(e)}")
        # Fallback: Route to support if the planner crashes
        return [SubTask(agent_target="customer_support", instruction="The orchestrator failed to parse the user's request. Please ask them to clarify.")]