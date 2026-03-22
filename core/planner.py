import os
import yaml
import json
import logging
from typing import List

from core.llm import call_llm
from models.llm_schemas import StandardLLMResponse
from models.state import AgentRequest, ExecutionPlan, Stage, Task, FailurePolicy
from core.agents.agent_registry import swarm_registry
from pydantic import BaseModel

logger = logging.getLogger("AgenticCore.Planner")

# Schema just for the LLM output. Its subset of 'Task'
# Required as we don't want to send all task fields to LLM, to prevent it from hallunicating
class LLMTaskSchema(BaseModel):
    agent_target: str
    instruction: str

class LLMPlanSchema(BaseModel):
    tasks: List[LLMTaskSchema]

def _load_yaml_workflow(workflow_name: str) -> ExecutionPlan:
    """Reads a Declarative YAML file and maps it to the Hierarchical Tree."""
    file_path = f"core/workflows/{workflow_name}.yaml"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Workflow template '{workflow_name}.yaml' not found in core/workflows/")

    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)

    planned_stages = []

    for stage_idx, stage_data in enumerate(data.get("stages", [])):
        tasks = []
        for t in stage_data.get("tasks", []):
            tasks.append(Task(
                agent_target=t["target_agent"],
                instruction=t["instruction"],
                status="pending",
                on_failure=FailurePolicy(t.get("on_failure", "TERMINATE").upper()) 
            ))

        planned_stages.append(Stage(
            stage_id=stage_idx,
            stage_name=stage_data.get("stage_name", f"Stage {stage_idx}"),
            tasks=tasks
        ))

    return ExecutionPlan(planned_stages=planned_stages)

async def _generate_llm_plan(request: AgentRequest) -> ExecutionPlan:
    """Calls the Master Planner LLM and maps the output to sequential stages."""
    planner_def = swarm_registry.get_agent("planner")

    available_agents = [f"- {name}: {agent.config.description}" for name, agent in swarm_registry.agents.items() if name not in ["planner", "evaluator"]]
    roster_string = "\n".join(available_agents)

    system_prompt = planner_def.system_prompt_builder(roster_string=roster_string)
    schema_json = LLMPlanSchema.model_json_schema()

    messages = [
        {"role": "system", "content": system_prompt, "cache_control": True},
        {"role": "user", "content": f"USER REQUEST: {request.user_prompt}"}
    ]

    try:
        response: StandardLLMResponse = await call_llm(
            messages=messages,
            response_schema=schema_json,
            tier=planner_def.config.llm_tier,
            temperature=planner_def.config.temperature,
            trace_id=request.thread_id
        )

        parsed_plan = LLMPlanSchema.model_validate_json(response.content)
        planned_stages = []

        for stage_idx, t in enumerate(parsed_plan.tasks):
            task_obj = Task(
                agent_target=t.agent_target,
                instruction=t.instruction,
                status="pending",
                on_failure=FailurePolicy.TERMINATE
            )
            planned_stages.append(Stage(
                stage_id=stage_idx,
                stage_name=f"Generated Step {stage_idx + 1}",
                tasks=[task_obj]
            ))

        return ExecutionPlan(planned_stages=planned_stages)

    except Exception as e:
        logger.error(f"[{request.thread_id}] Planner failed to generate DAG: {e}")
        fallback_task = Task(
            agent_target="support_agent", 
            instruction="The system failed to parse a complex plan. Please assist the user generally."
        )
        return ExecutionPlan(planned_stages=[Stage(stage_id=0, tasks=[fallback_task])])

async def get_execution_plan(request: AgentRequest) -> ExecutionPlan:
    """The Single Entry Point for creating an Execution Tree."""
    if request.workflow_name:
        logger.info(f"[{request.thread_id}] Loading declarative workflow: {request.workflow_name}")
        return _load_yaml_workflow(request.workflow_name)
    else:
        logger.info(f"[{request.thread_id}] Generating generative DAG via Planner LLM...")
        return await _generate_llm_plan(request)