from pydantic import BaseModel, Field
from typing import Optional

class BorrowerSWOTAnalysis(BaseModel):
    swot_analysis: Optional[str] = Field(None, description="SWOT Analysis")
