from pydantic import BaseModel, Field
from typing import Optional

class BusinessModelAnalysis(BaseModel):
    competitors: Optional[str] = Field(None, description="Content")
    suppliers: Optional[str] = Field(None, description="Suppliers")
    customers: Optional[str] = Field(None, description="Customers")
    strategy: Optional[str] = Field(None, description="Strategy")
