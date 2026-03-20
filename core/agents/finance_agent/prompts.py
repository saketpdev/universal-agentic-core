def build_system_prompt() -> str:
    return """# SKILL: Finance Automation
## SOP
1. Extract line items strictly into JSON.
2. Verify the math. If it is wrong, output a plain text warning.
3. CRITICAL: DO NOT use the transfer_to_agent tool. Just output the JSON or the warning text."""