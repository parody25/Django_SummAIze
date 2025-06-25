from pydantic import BaseModel, Field
from typing import Optional

from .borrower_details import BorrowerDetails
from .ownership import Ownership
from .loan_purpose_and_structure import LoanPurposeAndStructure
from .borrower_history_and_background import BorrowerHistoryAndBackground
from .industry_analysis import IndustryAnalysis
from .business_analysis import BusinessModelAnalysis
from .management_analysis import ManagementAnalysis
from .financial_analysis import FinancialAnalysis
from .security import Security
from .covenants_and_conditions import CovenantsAndConditions
from .policy_exceptions import PolicyExceptions
from .environmental_comments import EnvironmentalComments
from .risk_analysis import RiskAnalysis
from .peer_benchmarking_analysis import PeerBenchmarkingAnalysis
from .borrower_swot_analysis import BorrowerSWOTAnalysis
from .conclusion_and_recommendation import ConclusionAndRecommendation

class CreditApplication(BaseModel):
    credit_application_name: Optional[str] = Field(None, description="Credit Application Name")
    prepared_by: Optional[str] = Field(default="John Doe", description="Prepared by")
    date: Optional[str] = Field(None, description="Date prepared")
    borrower_details: BorrowerDetails = Field(default_factory=BorrowerDetails, description="Borrower Details")
    ownership: Ownership = Field(default_factory=Ownership, description="Ownership")
    loan_purpose_and_structure: LoanPurposeAndStructure = Field(default_factory=LoanPurposeAndStructure, description="Loan Purpose and Structure")
    borrower_history_and_background: BorrowerHistoryAndBackground = Field(default_factory=BorrowerHistoryAndBackground, description="Borrower History and Background")
    industry_analysis: IndustryAnalysis = Field(default_factory=IndustryAnalysis, description="Industry Analysis")
    business_model_analysis: BusinessModelAnalysis = Field(default_factory=BusinessModelAnalysis, description="Business Analysis")
    management_analysis: ManagementAnalysis = Field(default_factory=ManagementAnalysis, description="Management Analysis")
    financial_analysis: FinancialAnalysis = Field(default_factory=FinancialAnalysis, description="Financial Analysis")
    security: Security = Field(default_factory=Security, description="Security")
    covenants_and_conditions: CovenantsAndConditions = Field(default_factory=CovenantsAndConditions, description="Covenants and Conditions")
    policy_exceptions: PolicyExceptions = Field(default_factory=PolicyExceptions, description="Policy Exceptions")
    environmental_comments: EnvironmentalComments = Field(default_factory=EnvironmentalComments, description="Environmental Comments")
    risk_analysis: RiskAnalysis = Field(default_factory=RiskAnalysis, description="Risk Analysis")
    peer_benchmarking_analysis: PeerBenchmarkingAnalysis = Field(default_factory=PeerBenchmarkingAnalysis, description="Peer Benchmarking Analysis")
    borrower_swot_analysis: BorrowerSWOTAnalysis = Field(default_factory=BorrowerSWOTAnalysis, description="Borrower SWOT Analysis")
    conclusion_and_recommendation: ConclusionAndRecommendation = Field(default_factory=ConclusionAndRecommendation, description="Conclusion and Recommendation")
