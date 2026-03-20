import os
import logging
from typing import List, Dict, Any, Optional
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from openai import OpenAI
from dotenv import load_dotenv

from core.model_registry import MODEL_REGISTRY
from models.llm_schemas import StandardLLMResponse, StandardToolCall, StandardTokenUsage
from core.telemetry import TelemetryLogger

logger = logging.getLogger("AgenticCore.LLM")
load_dotenv()

# --- TRANSIENT ERROR LOGIC ---
class TransientAPIError(Exception):
    """Raised specifically for 429 Rate Limits or 5xx Server Errors."""
    pass

def _is_transient(exception: Exception) -> bool:
    """Helper to decide if we should retry based on the error message."""
    err_str = str(exception).lower()
    return any(keyword in err_str for keyword in ["rate limit", "429", "502", "503", "504", "timeout", "connection reset"])
# ----------------------------------

_openai_clients = {}

def _get_openai_client(base_url: str, api_key: str) -> OpenAI:
    if base_url not in _openai_clients:
        _openai_clients[base_url] = OpenAI(base_url=base_url, api_key=api_key)
    return _openai_clients[base_url]

def _execute_openai_compatible(config: Dict, messages: List[Dict], tools: Optional[List[Dict]], response_schema: Optional[Dict], api_key: str, temperature: float) -> StandardLLMResponse:
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
        "temperature": temperature  # <-- NOW DYNAMICALLY PASSED
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

def _execute_anthropic(config: Dict, messages: List[Dict], tools: Optional[List[Dict]], response_schema: Optional[Dict], api_key: str, temperature: float) -> StandardLLMResponse:
    """Handles Anthropic Claude. Uses Explicit Ephemeral Caching."""
    logger.info("Anthropic adapter triggered. Translating Explicit Cache markers...")

    system_prompt = ""
    anthropic_messages = []

    for m in messages:
        if m["role"] == "system":
            if m.get("cache_control"):
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
            clean_msg = {k: v for k, v in m.items() if k != "cache_control"}
            anthropic_messages.append(clean_msg)

    raise NotImplementedError("Anthropic message translation complete, but SDK execution is pending.")

# 🚀 The Tenacity Decorator: for exponential backoff
@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10), # Waits 2s, 4s, 8s, 10s...
    stop=stop_after_attempt(5),
    retry=retry_if_exception_type(TransientAPIError),
    reraise=True
)
def call_llm(
    messages: List[Dict], 
    tools: Optional[List[Dict]] = None, 
    response_schema: Optional[Dict] = None,
    tier: str = "worker",
    trace_id: Optional[str] = None,
    temperature: Optional[float] = None  # <-- NEW PARAMETER ADDED
) -> StandardLLMResponse:
    """The Universal Gateway: Routes requests based on provider strategy."""

    if tier not in MODEL_REGISTRY:
        logger.warning(f"Tier '{tier}' not found. Falling back to 'worker'.")
        tier = "worker"

    config = MODEL_REGISTRY[tier]
    api_key = os.environ.get(config["api_key_env"])

    if not api_key:
        raise ValueError(f"CRITICAL: Missing API key for tier '{tier}'.")

    # If temperature isn't explicitly passed via YAML/kwargs, fallback to model config
    validated_temperature: float = float(temperature if temperature is not None else config.get("default_temp", 0.1))

    logger.info(f"LLM [{tier.upper()}]: Routing to {config['model']} via {config['provider']} (Temp: {validated_temperature})")

    try:
        # 1. Execute Strategy Pattern Routing
        if config["provider"] in ["openai", "groq", "vllm", "deepseek"]:
            response = _execute_openai_compatible(config, messages, tools, response_schema, api_key, validated_temperature)
        elif config["provider"] == "anthropic":
            response = _execute_anthropic(config, messages, tools, response_schema, api_key, validated_temperature)
        else:
            raise ValueError(f"Unsupported provider: {config['provider']}")

        # 2. FINOPS MATH: Calculate actual USD cost
        if response.usage:
            p_cost = (response.usage.prompt_tokens / 1_000_000.0) * config.get("input_cost_per_m", 0.0)
            c_cost = (response.usage.completion_tokens / 1_000_000.0) * config.get("output_cost_per_m", 0.0)

            response.usage.prompt_cost_usd = p_cost
            response.usage.completion_cost_usd = c_cost
            response.usage.total_cost_usd = p_cost + c_cost

        return response

    except Exception as e:
        # 3. TELEMETRY & RETRY ROUTING
        if _is_transient(e):
            logger.warning(f"LLM Transient Error ({tier}): {str(e)}. Tenacity will backoff and retry...")
            if trace_id:
                telemetry = TelemetryLogger(trace_id=trace_id)
                telemetry.log_decision(
                    agent_id=f"llm_gateway_{tier}",
                    reasoning=f"Transient API error: {str(e)}. Triggering exponential backoff.",
                    context="Transient Failure Retry"
                )
            raise TransientAPIError(f"Network/Rate Limit: {str(e)}")
        else:
            logger.error(f"LLM Fatal Execution Failed on tier '{tier}': {str(e)}")
            if trace_id:
                telemetry = TelemetryLogger(trace_id=trace_id)
                telemetry.log_decision(
                    agent_id=f"llm_gateway_{tier}",
                    reasoning=f"Terminal API error: {str(e)}. Aborting LLM call.",
                    context="Terminal LLM Failure"
                )
            raise e # Instantly kills the run (e.g., bad API key, invalid schema)