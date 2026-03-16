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

def get_agent_context(agent_name: str) -> Dict[str, Any]:
    """Directly loads a specific agent's SOP and Judge for Swarm handoffs."""
    registry = load_registry()

    # Fallback safety
    if agent_name not in registry:
        # STRICT DAG FAIL:
        raise ValueError(f"CRITICAL: Agent '{agent_name}' not found in registry. The Planner hallucinated.")

    logger.info(f"Orchestrator: Loading context for '{agent_name}' agent.")
    skill_config = registry[agent_name]

    schema_class = get_schema_class(
        skill_config["evaluator_schema_module"], 
        skill_config["evaluator_schema_class"]
    )

    return {
        "agent_name": agent_name,
        "generator_prompt": skill_config["generator_prompt"],
        "evaluator_prompt": skill_config["evaluator_prompt"],
        "evaluator_schema": schema_class
    }