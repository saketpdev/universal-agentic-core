from pydantic import Field
from models.evaluations.base import BaseEvaluationSchema

class ComplianceEvaluationSchema(BaseEvaluationSchema):
    policy_adherence: float = Field(description="Score from 1.0 to 5.0 on adherence to company policy.")
    tone_professional: bool = Field(description="True if tone is de-escalating and professional.")