import logging

logger = logging.getLogger("AgenticCore.Security")

# PILLAR 6: Untrusted Content Boundaries
def handle_external_content(content: str) -> str:
    """Wraps inbound external data to prevent prompt injection."""
    return f"<untrusted_data>\n{content}\n</untrusted_data>"

# SPEC 03: Asymmetric Action Gate
def asymmetric_action_gate(user_prompt: str, requested_action: str) -> tuple[bool, str]:
    """
    Uses a fast screener to extract intent and cross-references it with the requested action.
    """
    # MOCK FAST SCREENER: In production, this is a separate, cheap LLM call (e.g., GPT-3.5)
    extracted_intent = "read"

    # Simulating the screener detecting a malicious or frustrated intent
    if "delete" in user_prompt.lower() or "cancel" in user_prompt.lower():
        extracted_intent = "delete"

    # Cross-Reference Policy Layer
    # If the user says "delete my account" but the LLM tries to run "fetch_user_data",
    # the LLM might have been tricked by a hidden prompt. Block it.
    if extracted_intent == "delete" and requested_action != "delete_record":
        logger.warning("INJECTION DETECTED: Intent/Action mismatch.")
        return False, "Action Blocked: Prompt injection detected."

    return True, "Authorized"