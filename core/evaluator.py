import os
import json
import logging
from typing import Type
from pydantic import BaseModel
from openai import OpenAI

logger = logging.getLogger("AgenticCore.Evaluator")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY")
)

def run_dynamic_evaluation(output_text: str, objective: str, evaluator_prompt: str, schema_class: Type[BaseModel]) -> BaseModel:
    """A truly universal Judge that adapts its criteria based on the loaded skill."""
    
    schema_json = schema_class.model_json_schema()
    
    messages = [
        {
            "role": "system", 
            "content": f"{evaluator_prompt}\nYou MUST return valid JSON matching this exact schema: {json.dumps(schema_json)}"
        },
        {
            "role": "user", 
            "content": f"OBJECTIVE: {objective}\n\nOUTPUT TO GRADE:\n{output_text}"
        }
    ]
    
    logger.info(f"Running dynamic Evaluator using {schema_class.__name__} schema...")
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        
        result_dict = json.loads(response.choices[0].message.content)
        return schema_class(**result_dict)
        
    except Exception as e:
        logger.error(f"Evaluator failed to parse JSON: {e}")
        return schema_class(
            reasoning="System parsing error.",
            pass_status=False,
            critique="CRITICAL SYSTEM ERROR: Evaluator failed to parse output. You must rewrite and correct your formatting."
        )