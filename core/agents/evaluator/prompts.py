def build_system_prompt(rubric: str, schema_json_str: str) -> str:
    """Builds the dynamic system prompt for the Evaluator."""
    
    return f"""{rubric}

# STRICT OUTPUT REQUIREMENT
You MUST return valid JSON matching this exact schema:
{schema_json_str}
"""