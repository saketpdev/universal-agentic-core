import json
import logging
from typing import Type, Optional
from pydantic import BaseModel
from core.llm import call_llm
from core.agents.agent_registry import swarm_registry

logger = logging.getLogger("AgenticCore.Evaluator")

def run_dynamic_evaluation(
    output_text: str, 
    objective: str, 
    evaluator_prompt: str, 
    schema_class: Type[BaseModel],
    trace_id: Optional[str] = None
) -> BaseModel:
    """A truly universal Judge routed through the Declarative Swarm Registry."""
    
    schema_json = schema_class.model_json_schema()
    
    # 1. Pull the Evaluator config from the Registry
    evaluator_def = swarm_registry.get_agent("evaluator")
    
    # 2. Build the dynamic prompt
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
        # 3. Execute the LLM using the YAML configs AND the trace_id!
        response_msg = call_llm(
            messages=messages,
            response_schema=schema_json,
            tier=evaluator_def.config.llm_tier,            # <-- Pulled from config.yaml
            temperature=evaluator_def.config.temperature,  # <-- Pulled from config.yaml
            trace_id=trace_id                              # <-- Passed to OTel!
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