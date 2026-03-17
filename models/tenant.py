from pydantic import BaseModel, Field
from uuid import UUID

class TenantBaggage(BaseModel):
    """
    Strict schema for OpenTelemetry Baggage.
    Enforces Rule 1: Only Opaque Identifiers allowed in network headers.
    """
    tenant_id: UUID  # Strict UUID validation. Will reject "AcmeCorp"
    billing_tier: str = Field(default="standard")

    def to_dict(self) -> dict[str, str]:
        """Serializes safely for OpenTelemetry Baggage injection."""
        return {
            "tenant_id": str(self.tenant_id),
            "billing_tier": self.billing_tier
        }