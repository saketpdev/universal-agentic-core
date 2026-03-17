from typing import Dict, Any

# The LLM Gateway Configuration
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "planner": {
        "provider": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "supports_json_schema": False,
        "default_temp": 0.0,
        # FinOps Pricing (USD per 1M tokens)
        "input_cost_per_m": 0.59,
        "output_cost_per_m": 0.79
    },
    "worker": {
        "provider": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.1-8b-instant", 
        "supports_json_schema": False,
        "default_temp": 0.1,
        # FinOps Pricing (USD per 1M tokens) - Blindingly cheap
        "input_cost_per_m": 0.05,  
        "output_cost_per_m": 0.08
    },
    "judge": {
        "provider": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "supports_json_schema": False,
        "default_temp": 0.0,
        # FinOps Pricing (USD per 1M tokens)
        "input_cost_per_m": 0.59,
        "output_cost_per_m": 0.79
    }
}