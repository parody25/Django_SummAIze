from pydantic import BaseModel, Field
from typing import Optional

class Security(BaseModel):
    personal_guarantee : Optional[str] = Field(None, description="Personal Guarantee")
    real_estate_security : Optional[str] = Field(None, description="Real Estate Security")
    equipment_security : Optional[str] = Field(None, description="Equipment Security")
    inventory_and_accounts_receivable_security : Optional[str] = Field(None, description="Inventory and Accounts Receivable Security ")