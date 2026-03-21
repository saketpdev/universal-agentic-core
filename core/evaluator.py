import json
import logging
from typing import Type, Optional

from models.evaluations.base import BaseEvaluationSchema

from core.llm import call_llm
from core.agents.agent_registry import swarm_registry

logger = logging.getLogger("AgenticCore.Evaluator")

# 🚀 CHANGED TO ASYNC
async def run_dynamic_evaluation(
    output_text: str,
    objective: str,
    evaluator_prompt: str,
    schema_class: Type[BaseEvaluationSchema],
    trace_id: Optional[str] = None
) -> BaseEvaluationSchema:

    schema_json = schema_class.model_json_schema()
    evaluator_def = swarm_registry.get_agent("evaluator")

    system_prompt = evaluator_def.system_prompt_builder(
        rubric=evaluator_prompt,
        schema_json_str=json.dumps(schema_json)
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"OBJECTIVE: {objective}\n\nOUTPUT TO GRADE:\n{output_text}"}
    ]

    logger.info(f"Evaluator: Grading against {schema_class.__name__} schema...")

    try:
        response_msg = await call_llm(
            messages=messages,
            response_schema=schema_json,
            tier=evaluator_def.config.llm_tier,
            temperature=evaluator_def.config.temperature,
            trace_id=trace_id
        )

        result_dict = json.loads(response_msg.content)
        return schema_class(**result_dict)

    except Exception as e:
        logger.error(f"Evaluator failed to parse JSON: {e}")
        return schema_class(
            reasoning="System parsing error.",
            pass_status=False,
            critique="CRITICAL SYSTEM ERROR: Evaluator failed to parse output. You must rewrite and correct your formatting."
        )