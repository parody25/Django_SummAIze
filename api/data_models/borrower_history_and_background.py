from pydantic import BaseModel, Field
from typing import Optional

class BorrowerHistoryAndBackground(BaseModel):
    borrower_profile: Optional[str] = Field(None, description="Content")
