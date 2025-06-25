from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import date

class Owner(BaseModel):
    name: Optional[str] = Field(None, description="Name")
    position: Optional[str] = Field(None, description="Position")
    DOB: Optional[str] = Field(None, description="DOB")
    percentage_ownership: Optional[str] = Field(None, description="Percentage Ownership")
    net_worth: Optional[str] = Field(None, description="Net Worth")

class Ownership(BaseModel):
    owners: Optional[List[Owner]] = Field(default_factory=lambda: [Owner()], description="Owners")
