from pydantic import BaseModel, Field
from typing import Optional

class EnvironmentalComments(BaseModel):
    field_visit_details: Optional[str] = Field(None, description="Field visit details")
    environmental_considerations: Optional[str] = Field(None, description="Environmental considerations")
