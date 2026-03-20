import json
import logging
from typing import List

from core.llm import call_llm
from models.llm_schemas import StandardLLMResponse
from models.state import AgentRequest, ExecutionPlan, SubTask
from core.agents.agent_registry import swarm_registry
from pydantic import BaseModel

logger = logging.getLogger("AgenticCore.Planner")

class DAGSchema(BaseModel):
    tasks: List[SubTask]

def generate_dag(request: AgentRequest) -> ExecutionPlan:
    """Generates the DAG execution plan dynamically using the Declarative Swarm Registry."""
    logger.info(f"[{request.thread_id}] Generating DAG...")

    planner_def = swarm_registry.get_agent("planner")

    # --- DYNAMIC ROSTER GENERATION ---
    available_agents = []
    for name, agent in swarm_registry.agents.items():
        if name not in ["planner", "evaluator"]:
            available_agents.append(f"- {name}: {agent.config.description}")

    roster_string = "\n".join(available_agents)

    system_prompt = planner_def.system_prompt_builder(roster_string=roster_string)

    schema_json = DAGSchema.model_json_schema()

    messages = [
        {"role": "system", "content": system_prompt, "cache_control": True},
        {"role": "user", "content": f"USER REQUEST: {request.user_prompt}"}
    ]

    try:
        response: StandardLLMResponse = call_llm(
            messages=messages,
            response_schema=schema_json,
            tier=planner_def.config.llm_tier,
            temperature=planner_def.config.temperature,
            trace_id=request.thread_id
        )

        parsed_plan = json.loads(response.content)

        return ExecutionPlan(
            tasks=parsed_plan.get("tasks", [])
        )

    except Exception as e:
        logger.error(f"[{request.thread_id}] Planner failed to generate DAG: {e}")

        return ExecutionPlan(
            tasks=[
                SubTask(
                    agent_target="support_agent",
                    instruction=f"The system failed to parse a complex plan for: '{request.user_prompt}'. Please assist the user generally.",
                    status="pending"
                )
            ]
        )