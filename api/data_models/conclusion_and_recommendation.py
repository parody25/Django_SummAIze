from pydantic import BaseModel, Field
from typing import Optional

class ConclusionAndRecommendation(BaseModel):
    justification_for_loan: Optional[str] = Field(None, description="Justification for Loan")
    relationship_managers_comments: Optional[str] = Field(None, description="Relationship Managers Comments")
