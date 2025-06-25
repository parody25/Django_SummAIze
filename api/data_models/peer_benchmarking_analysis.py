from pydantic import BaseModel, Field
from typing import Optional

class FinancialMetrics(BaseModel):
    sales: Optional[str] = Field(None, description="Sales $")
    net_profit: Optional[str] = Field(None, description="Net profit $")
    owners_equity: Optional[str] = Field(None, description="Owners’ equity $")
    working_capital: Optional[str] = Field(None, description="Working capital $")
    working_capital_ratio: Optional[str] = Field(None, description="Working capital ratio")
    operating_cash_flow: Optional[str] = Field(None, description="Operating cash flow $")
    funded_debt_ebitda_ratio: Optional[str] = Field(None, description="Funded debt/EBITDA Ratio")
    debt_service_coverage_ratio: Optional[str] = Field(None, description="Debt service coverage ratio")
    ar_turnaround_days: Optional[str] = Field(None, description="A/R turnaround (days)")
    inventory_turnaround_days: Optional[str] = Field(None, description="Inventory turnaround (days)")
    payables_turnaround_days: Optional[str] = Field(None, description="Payables turnaround (days)")

class PeerBenchmarkingAnalysis(BaseModel):
    observations: Optional[str] = Field(None, description="Observations")
    year_ended_2019: Optional[FinancialMetrics] = Field(None, description="Year Ended 2019")
    year_ended_2018: Optional[FinancialMetrics] = Field(None, description="Year Ended 2018")
    year_ended_2017: Optional[FinancialMetrics] = Field(None, description="Year Ended 2017")
    industry_benchmarks: Optional[FinancialMetrics] = Field(None, description="Industry Benchmarks")
