from pydantic import BaseModel, Field
from typing import Optional

class PolicyExceptions(BaseModel):
    policy_exceptions: Optional[str] = Field(None, description="policy_exceptions")
