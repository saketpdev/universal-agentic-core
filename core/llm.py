import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
from core.model_registry import MODEL_REGISTRY

logger = logging.getLogger("AgenticCore.LLM")
load_dotenv()

# Cache clients so we don't constantly spin up new connections if base URLs differ
_clients = {}

def get_client(base_url: str, api_key: str) -> OpenAI:
    if base_url not in _clients:
        _clients[base_url] = OpenAI(base_url=base_url, api_key=api_key)
    return _clients[base_url]

def call_llm(
    messages: List[Dict], 
    tools: Optional[List[Dict]] = None, 
    response_schema: Optional[Dict] = None,
    tier: str = "worker"
) -> Any:
    """Universal LLM Gateway supporting Multiplexing and Vendor Quirks."""
    
    if tier not in MODEL_REGISTRY:
        logger.warning(f"Tier '{tier}' not found. Falling back to 'worker'.")
        tier = "worker"
        
    config = MODEL_REGISTRY[tier]
    api_key = os.environ.get(config["api_key_env"])
    
    if not api_key:
        raise ValueError(f"CRITICAL: Missing API key for tier '{tier}'. Set {config['api_key_env']}")

    client = get_client(config["base_url"], api_key)
    logger.info(f"LLM [{tier.upper()}]: Routing to {config['model']} via {config['provider']}")
    
    kwargs = {
        "model": config["model"],
        "messages": messages,
        "temperature": config.get("default_temp", 0.1)
    }
    
    if tools:
        kwargs["tools"] = tools

    if response_schema:
        if config["supports_json_schema"]:
            logger.info("LLM: Vendor supports strict json_schema. Enforcing.")
            kwargs["response_format"] = {
                "type": "json_schema", 
                "json_schema": {"name": "structured_output", "schema": response_schema, "strict": True}
            }
        else:
            logger.info("LLM: Vendor requires json_object fallback.")
            kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message
    except Exception as e:
        logger.error(f"LLM API Call failed on tier '{tier}': {str(e)}")
        raise e