import json
import logging
from typing import Type
from pydantic import BaseModel
from core.llm import call_llm

logger = logging.getLogger("AgenticCore.Evaluator")

def run_dynamic_evaluation(output_text: str, objective: str, evaluator_prompt: str, schema_class: Type[BaseModel]) -> BaseModel:
    """A truly universal Judge routed through the LLM Gateway."""
    
    schema_json = schema_class.model_json_schema()
    
    messages = [
        {
            "role": "system", 
            "content": f"{evaluator_prompt}\nYou MUST return valid JSON matching this exact schema:\n{json.dumps(schema_json)}"
        },
        {
            "role": "user", 
            "content": f"OBJECTIVE: {objective}\n\nOUTPUT TO GRADE:\n{output_text}"
        }
    ]
    
    logger.info(f"Evaluator: Grading against {schema_class.__name__} schema...")
    
    try:
        response_msg = call_llm(
            messages=messages,
            response_schema=schema_json,
            tier="judge"
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