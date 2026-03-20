from pydantic import Field
from models.evaluations.base import BaseEvaluationSchema

class FinanceEvaluationSchema(BaseEvaluationSchema):
    source_stated_total: float = Field(
        description="The total amount explicitly stated by the user in the ORIGINAL objective text."
    )
    calculated_line_items_total: float = Field(
        description="The actual mathematical sum of the line items provided."
    )
    discrepancy_detected: bool = Field(
        description="True ONLY if the source_stated_total does not equal the calculated_line_items_total."
    )