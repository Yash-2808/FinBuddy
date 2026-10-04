"""Pydantic schemas for FastAPI endpoints."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="User financial question or instruction")
    chat_history: Optional[List[Dict[str, str]]] = Field(default=[], description="List of previous conversation turns")


class ChatResponse(BaseModel):
    response: str
    tool_calls: List[Dict[str, Any]] = []
    model_used: str


class CreditApplicantSchema(BaseModel):
    person_age: int = Field(35, ge=18, le=100)
    person_income: float = Field(75000.0, ge=1000)
    person_emp_length: float = Field(5.0, ge=0)
    loan_amnt: float = Field(15000.0, ge=500)
    loan_int_rate: float = Field(11.5, ge=1.0, le=40.0)
    loan_percent_income: Optional[float] = None
    cb_person_cred_hist_length: float = Field(7.0, ge=0)
    person_home_ownership: str = Field("RENT", description="RENT, OWN, MORTGAGE, OTHER")
    loan_intent: str = Field("PERSONAL", description="PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION")
    loan_grade: str = Field("B", description="A, B, C, D, E, F, G")
    cb_person_default_on_file: str = Field("N", description="Y or N")
    credit_score: Optional[int] = Field(710, ge=300, le=850)
    debt_to_income_ratio: Optional[float] = Field(0.28, ge=0.0, le=1.5)
    previous_defaults_count: Optional[int] = Field(0, ge=0, le=10)


class FeatureContribution(BaseModel):
    feature: str
    value: float
    shap_value: float
    impact: str
    abs_importance: float


class CreditScoreResponse(BaseModel):
    default_probability: float
    approval_score: float
    risk_tier: str
    recommendation: str
    optimal_threshold: Optional[float] = 0.6903
    base_value: float
    narrative_explanation: str
    feature_contributions: List[FeatureContribution]
    top_risk_drivers: List[FeatureContribution]
    top_protective_factors: List[FeatureContribution]


class DriftCheckRequest(BaseModel):
    drift_factor: float = Field(0.0, ge=0.0, le=2.0, description="Macroeconomic stress level to simulate drift")
    sample_size: int = Field(1000, ge=100, le=10000)


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    explainer_loaded: bool
    features: List[str]
