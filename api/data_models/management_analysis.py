from pydantic import BaseModel, Field
from typing import Optional

class ManagementAnalysis(BaseModel):
    board_of_directors_profile: Optional[str] = Field(None, description="Board of Directors Profile")
