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

    # Strip internal 'cache_control' flags so the strict OpenAI SDK doesn't crash
    clean_messages = []
    for m in messages:
        clean_msg = {k: v for k, v in m.items() if k != "cache_control"}
        clean_messages.append(clean_msg)

    kwargs = {
        "model": config["model"],
        "messages": clean_messages,
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
    """Handles Anthropic Claude. Uses Explicit Ephemeral Caching."""
    logger.info("Anthropic adapter triggered. Translating Explicit Cache markers...")
    
    system_prompt = ""
    anthropic_messages = []
    
    for m in messages:
        if m["role"] == "system":
            # Anthropic handles the System Prompt completely separately from the messages array
            if m.get("cache_control"):
                # 🚨 ANTHROPIC EXPLICIT CACHE INJECTION 🚨
                system_prompt = [
                    {
                        "type": "text", 
                        "text": m["content"], 
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            else:
                system_prompt = m["content"]
        else:
            # Strip internal flags for standard messages
            clean_msg = {k: v for k, v in m.items() if k != "cache_control"}
            anthropic_messages.append(clean_msg)

    # Here is where you will eventually call the Anthropic SDK:
    # client.messages.create(system=system_prompt, messages=anthropic_messages, ...)
    raise NotImplementedError("Anthropic message translation complete, but SDK execution is pending.")

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
        # 1. Execute Strategy Pattern Routing
        if config["provider"] in ["openai", "groq", "vllm", "deepseek"]:
            response = _execute_openai_compatible(config, messages, tools, response_schema, api_key)
        elif config["provider"] == "anthropic":
            response = _execute_anthropic(config, messages, tools, response_schema, api_key)
        else:
            raise ValueError(f"Unsupported provider: {config['provider']}")
            
        # 2. FINOPS MATH: Calculate actual USD cost
        if response.usage:
            # Divide by 1,000,000 to get the multiplier, then multiply by the registry price
            p_cost = (response.usage.prompt_tokens / 1_000_000.0) * config.get("input_cost_per_m", 0.0)
            c_cost = (response.usage.completion_tokens / 1_000_000.0) * config.get("output_cost_per_m", 0.0)
            
            response.usage.prompt_cost_usd = p_cost
            response.usage.completion_cost_usd = c_cost
            response.usage.total_cost_usd = p_cost + c_cost
            
        return response
            
    except Exception as e:
        logger.error(f"LLM Gateway Execution Failed on tier '{tier}': {str(e)}")
        raise e