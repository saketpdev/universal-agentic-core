# This lives completely outside the core engine!
from pydantic import BaseModel, Field
from typing import Optional

class FinancePluginState(BaseModel):
    extracted_total: Optional[float] = None
    invoice_reference_id: Optional[str] = None
    math_discrepancy_flag: bool = False