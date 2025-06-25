from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import date

class BorrowerDetails(BaseModel):
    borrower_name: Optional[str] = Field(None, description="Borrower Name")
    date_of_application: Optional[str] = Field(None, description="Date of Application")
    type_of_business: Optional[str] = Field(None, description="Type of Business")
    borrower_risk_rating: Optional[str] = Field(None, description="Risk Rating")
    new_or_existing: Optional[str] = Field(None, description="New or Existing")
    naics_code: Optional[str] = Field(None, description="NAICS Code")
    borrower_address: Optional[str] = Field(None, description="Borrower Address")
    telephone: Optional[str] = Field(None, description="Telephone")
    email_address: Optional[str] = Field(None, description="Email Address")
    fax_number: Optional[str] = Field(None, description="Fax Number")
    branch_number: Optional[str] = Field(None, description="Branch Number")
    account_number: Optional[str] = Field(None, description="Account Number")
