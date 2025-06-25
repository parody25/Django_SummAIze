from pydantic import BaseModel, Field
from typing import Optional

class RiskAnalysis(BaseModel):
    risk_rating: Optional[str] = Field(None, description="Risk rating")
