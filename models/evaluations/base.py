from pydantic import BaseModel, Field

class BaseEvaluation(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning before grading.")
    pass_status: bool = Field(description="True if the output completely satisfies the objective and constraints. False otherwise.")
    critique: str = Field(description="Specific feedback for the generator if pass_status is False.")