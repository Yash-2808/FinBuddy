"""LangChain Tool for Underwriting and Credit Risk Scoring with SHAP Explainability."""

import json
from typing import Optional, Type, Dict, Any
from pydantic import BaseModel, Field

try:
    from langchain_core.tools import BaseTool
except ImportError:
    class BaseTool:
        name: str = ""
        description: str = ""
        def invoke(self, input_dict: Any) -> Any:
            if isinstance(input_dict, dict):
                return self._run(**input_dict)
            return self._run(input_dict)

from src.ml.explain import CreditRiskExplainer


class CreditRiskInput(BaseModel):
    person_age: int = Field(..., description="Age of the applicant in years (e.g. 35)")
    person_income: float = Field(..., description="Annual income in USD (e.g. 75000)")
    person_emp_length: float = Field(..., description="Years of employment (e.g. 5.0)")
    loan_amnt: float = Field(..., description="Requested loan amount in USD (e.g. 15000)")
    loan_int_rate: float = Field(..., description="Interest rate percentage (e.g. 11.5)")
    cb_person_cred_hist_length: float = Field(..., description="Credit history length in years (e.g. 7.0)")
    person_home_ownership: str = Field("RENT", description="Home status: RENT, OWN, MORTGAGE, OTHER")
    loan_intent: str = Field("PERSONAL", description="Loan intent: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION")
    loan_grade: str = Field("B", description="Credit grade: A, B, C, D, E, F, G")
    cb_person_default_on_file: str = Field("N", description="Default history: Y or N")
    credit_score: Optional[int] = Field(710, description="FICO score (optional)")
    debt_to_income_ratio: Optional[float] = Field(0.28, description="DTI ratio (optional)")
    previous_defaults_count: Optional[int] = Field(0, description="Count of past defaults (optional)")
    loan_percent_income: Optional[float] = Field(
        None, description="Ratio of loan amount to annual income (calculated automatically if omitted)"
    )


class CreditRiskTool(BaseTool):
    name: str = "credit_risk_assessment_tool"
    description: str = (
        "Assesses credit risk, default probability, and underwrites loan applications using an XGBoost ML model "
        "and SHAP explainability. Returns default probability, underwriting recommendation, and top risk factors."
    )
    args_schema: Type[BaseModel] = CreditRiskInput
    
    def _run(
        self,
        person_age: int,
        person_income: float,
        person_emp_length: float,
        loan_amnt: float,
        loan_int_rate: float,
        cb_person_cred_hist_length: float,
        person_home_ownership: str = "RENT",
        loan_intent: str = "PERSONAL",
        loan_grade: str = "B",
        cb_person_default_on_file: str = "N",
        credit_score: int = 710,
        debt_to_income_ratio: float = 0.28,
        previous_defaults_count: int = 0,
        loan_percent_income: Optional[float] = None
    ) -> str:
        try:
            if loan_percent_income is None:
                loan_percent_income = round(loan_amnt / max(person_income, 1.0), 3)

            applicant = {
                "person_age": person_age,
                "person_income": person_income,
                "person_emp_length": person_emp_length,
                "loan_amnt": loan_amnt,
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
                "person_home_ownership": person_home_ownership,
                "loan_intent": loan_intent,
                "loan_grade": loan_grade,
                "cb_person_default_on_file": cb_person_default_on_file,
                "credit_score": credit_score,
                "debt_to_income_ratio": debt_to_income_ratio,
                "previous_defaults_count": previous_defaults_count,
            }

            explainer = CreditRiskExplainer()
            result = explainer.predict_and_explain(applicant)

            output = (
                f"### 📋 Credit Risk & Underwriting Assessment\n\n"
                f"- **Default Probability**: {result['default_probability']:.1%}\n"
                f"- **Approval Score**: {result['approval_score']} / 100\n"
                f"- **Risk Classification**: {result['risk_tier']}\n"
                f"- **Recommendation**: **{result['recommendation']}**\n\n"
                f"**Executive Narrative**:\n{result['narrative_explanation']}\n\n"
                f"**Top Risk-Increasing Drivers (SHAP)**:\n"
            )

            for d in result["top_risk_drivers"]:
                output += f"- `{d['feature']}` = {d['value']} (SHAP Impact: +{d['shap_value']})\n"

            output += "\n**Top Protective Factors (SHAP)**:\n"
            for p in result["top_protective_factors"]:
                output += f"- `{p['feature']}` = {p['value']} (SHAP Impact: {p['shap_value']})\n"

            return output

        except Exception as e:
            return f"Error executing Credit Risk Tool: {str(e)}"

    async def _arun(self, *args, **kwargs) -> str:
        return self._run(*args, **kwargs)
