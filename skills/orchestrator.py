import json
import os
import importlib
import logging
from typing import Dict, Any

logger = logging.getLogger("AgenticCore.Skills")

def load_registry() -> Dict[str, Any]:
    """Reads the configuration from disk."""
    registry_path = os.path.join(os.path.dirname(__file__), "registry.json")
    with open(registry_path, "r") as f:
        return json.load(f)

def get_schema_class(module_name: str, class_name: str):
    """Dynamically imports a Pydantic class by its string name."""
    try:
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to dynamically load schema {module_name}.{class_name}: {e}")
        raise

def load_skill_context(user_prompt: str) -> Dict[str, Any]:
    """Loads prompt text from JSON and the schema class from dynamic import."""
    registry = load_registry()
    prompt_lower = user_prompt.lower()
    
    selected_skill_key = "customer_support" # Default fallback
    
    # Dynamic keyword routing
    for skill_key, skill_data in registry.items():
        if any(keyword in prompt_lower for keyword in skill_data.get("trigger_keywords", [])):
            selected_skill_key = skill_key
            break
            
    logger.info(f"Orchestrator: Routing to {selected_skill_key} skill.")
    skill_config = registry[selected_skill_key]
    
    # Instantiate the dynamic class mapping
    schema_class = get_schema_class(
        skill_config["evaluator_schema_module"], 
        skill_config["evaluator_schema_class"]
    )
    
    return {
        "generator_prompt": skill_config["generator_prompt"],
        "evaluator_prompt": skill_config["evaluator_prompt"],
        "evaluator_schema": schema_class
    }