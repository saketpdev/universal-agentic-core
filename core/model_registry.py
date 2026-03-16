from typing import Dict, Any

# The LLM Gateway Configuration
MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "planner": {
        "provider": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "supports_json_schema": False, # Groq's current quirk
        "default_temp": 0.0
    },
    "worker": {
        "provider": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.1-8b-instant", # 10x cheaper and faster for ReAct loops
        "supports_json_schema": False,
        "default_temp": 0.1
    },
    "judge": {
        "provider": "groq",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile", # Needs high IQ to evaluate outputs
        "supports_json_schema": False,
        "default_temp": 0.0
    }
}