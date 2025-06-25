from pydantic import BaseModel, Field
from typing import Optional

class FinancialAnalysis(BaseModel):
    ratios: Optional[str] = Field(None, description="Financial ratios")
    analysis: Optional[str] = Field(None, description="Financial analysis")
    
