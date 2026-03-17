from pydantic import BaseModel
from typing import List, Optional, Any, Dict

class StandardToolCall(BaseModel):
    id: str
    function_name: str
    arguments: str # JSON string of arguments

class StandardTokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cost_usd: float = 0.0
    completion_cost_usd: float = 0.0
    total_cost_usd: float = 0.0

class StandardLLMResponse(BaseModel):
    content: str
    tool_calls: List[StandardToolCall] = []
    usage: StandardTokenUsage
    raw_provider_response: Any = None # Keep the original just in case for deep debugging