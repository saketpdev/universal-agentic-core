import os
import logging
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv

from core.model_registry import MODEL_REGISTRY
from models.llm_schemas import StandardLLMResponse, StandardToolCall, StandardTokenUsage

logger = logging.getLogger("AgenticCore.LLM")
load_dotenv()

_openai_clients = {}

def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
    if base_url not in _openai_clients:
        _openai_clients[base_url] = OpenAI(base_url=base_url, api_key=api_key)
    return _openai_clients[base_url]

def _execute_openai_compatible(config: Dict, messages: List[Dict], tools: Optional[List[Dict]], response_schema: Optional[Dict], api_key: str) -> StandardLLMResponse:
    """Handles OpenAI, Groq, DeepSeek, vLLM, and any OpenAI-compatible API."""
    client = _get_openai_client(config["base_url"], api_key)
    
    kwargs = {
        "model": config["model"],
        "messages": messages,
        "temperature": config.get("default_temp", 0.1)
    }
    
    if tools:
        kwargs["tools"] = tools

    if response_schema:
        if config["supports_json_schema"]:
            kwargs["response_format"] = {
                "type": "json_schema", 
                "json_schema": {"name": "structured_output", "schema": response_schema, "strict": True}
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    raw_msg = response.choices[0].message
    raw_usage = getattr(response, "usage", None)

    usage_model = StandardTokenUsage(
        prompt_tokens=raw_usage.prompt_tokens if raw_usage else 0,
        completion_tokens=raw_usage.completion_tokens if raw_usage else 0,
        total_tokens=raw_usage.total_tokens if raw_usage else 0
    )

    standard_tools = []
    if getattr(raw_msg, "tool_calls", None):
        for tc in raw_msg.tool_calls:
            standard_tools.append(StandardToolCall(
                id=tc.id,
                function_name=tc.function.name,
                arguments=tc.function.arguments
            ))

    return StandardLLMResponse(
        content=raw_msg.content or "",
        tool_calls=standard_tools,
        usage=usage_model,
        raw_provider_response=raw_msg
    )

def _execute_anthropic(config: Dict, messages: List[Dict], tools: Optional[List[Dict]], response_schema: Optional[Dict], api_key: str) -> StandardLLMResponse:
    """
    Placeholder adapter for Anthropic Claude.
    To implement: import anthropic, translate 'messages' roles, and map 'content_blocks'.
    """
    logger.info("Anthropic adapter triggered (Implementation pending anthropic SDK install).")
    # You will implement the Anthropic SDK mapping here, returning a StandardLLMResponse.
    raise NotImplementedError("Anthropic execution branch is configured but requires SDK logic.")

def call_llm(
    messages: List[Dict], 
    tools: Optional[List[Dict]] = None, 
    response_schema: Optional[Dict] = None,
    tier: str = "worker"
) -> StandardLLMResponse:
    """The Universal Gateway: Routes requests based on provider strategy."""
    
    if tier not in MODEL_REGISTRY:
        logger.warning(f"Tier '{tier}' not found. Falling back to 'worker'.")
        tier = "worker"
        
    config = MODEL_REGISTRY[tier]
    api_key = os.environ.get(config["api_key_env"])
    
    if not api_key:
        raise ValueError(f"CRITICAL: Missing API key for tier '{tier}'.")

    logger.info(f"LLM [{tier.upper()}]: Routing to {config['model']} via {config['provider']}")

    try:
        # Strategy Pattern Routing
        if config["provider"] in ["openai", "groq", "vllm", "deepseek"]:
            return _execute_openai_compatible(config, messages, tools, response_schema, api_key)
        elif config["provider"] == "anthropic":
            return _execute_anthropic(config, messages, tools, response_schema, api_key)
        else:
            raise ValueError(f"Unsupported provider: {config['provider']}")
            
    except Exception as e:
        logger.error(f"LLM Gateway Execution Failed on tier '{tier}': {str(e)}")
        raise e