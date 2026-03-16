import logging

logger = logging.getLogger("AgenticCore.Security")

# PILLAR 6: Untrusted Content Boundaries
def handle_external_content(content: str) -> str:
    """Wraps inbound external data to prevent prompt injection."""
    return f"<untrusted_data>\n{content}\n</untrusted_data>"

def asymmetric_action_gate(task_instruction: str, requested_action: str) -> tuple[bool, str]:
    """
    Evaluates the requested tool against the STRICT local task instruction.
    """
    # TODO (Roadmap): Replace this mock with a lightweight classifier (e.g., ONNX / Llama-3-8B)
    extracted_intent = "read"

    if "delete" in task_instruction.lower() or "cancel" in task_instruction.lower():
        extracted_intent = "delete"

    # Cross-Reference Policy Layer
    if extracted_intent == "delete" and requested_action != "delete_record":
        logger.warning("INJECTION DETECTED: Intent/Action mismatch.")
        return False, "Action Blocked: Prompt injection detected."

    return True, "Authorized"