from pydantic import BaseModel, Field
from typing import List, Optional

class Loan(BaseModel):
    loan_type: Optional[str] = Field(None, description="Loan Type")
    loan_term: Optional[str] = Field(None, description="Loan Term")
    max_loan_amount: Optional[str] = Field(None, description="Max Loan Amount")
    amortization: Optional[str] = Field(None, description="Amortization")
    payment_terms: Optional[str] = Field(None, description="Payment Terms")
    payment_amount: Optional[str] = Field(None, description="Payment Amount")
    interest_rate: Optional[str] = Field(None, description="Interest Rate")
    application_fees: Optional[str] = Field(None, description="Application Fees")
    group_and_relationship_profitability: Optional[str] = Field(None, description="Fees")

class LoanPurposeAndStructure(BaseModel):
    loan_purpose: Optional[str] = Field(None, description="Loan Purpose")
    loans: Optional[List[Loan]] = Field(default_factory= lambda: [Loan()], description="Loans")
