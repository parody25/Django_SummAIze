from pydantic import BaseModel, Field
from typing import Optional

class FinancialCovenants(BaseModel):
    debt_to_equity_ratio: Optional[str] = Field(None, description="Debt to Equity ratio")
    working_capital_ratio: Optional[str] = Field(None, description="Working capital (current)ratio")
    debt_service_coverage_ratio: Optional[str] = Field(None, description="Debt service coverage ratio")
    funded_debt_ebitda_ratio: Optional[str] = Field(None, description="Funded debt/EBITDA ratio")
    minimum_shareholder_equity: Optional[str] = Field(None, description="Minimum shareholder equity")

class CovenantsAndConditions(BaseModel):
    financial_covenants : Optional[FinancialCovenants] = Field(default_factory=FinancialCovenants, description="Financial covenants")
    reporting_covenants : Optional[str] = Field(None, description="Reporting Covenants")
    terms_and_conditions : Optional[str] = Field(None, description="Terms and Conditions")