from pydantic import BaseModel, Field
from typing import Optional

class IndustryAnalysis(BaseModel):
    politcal: Optional[str] = Field(None, description="Political")
    economic: Optional[str] = Field(None, description="Economic")
    social: Optional[str] = Field(None, description="Social")
    technological: Optional[str] = Field(None, description="Technological")
    environmental: Optional[str] = Field(None, description="Environmental")
    legal: Optional[str] = Field(None, description="Legal")
