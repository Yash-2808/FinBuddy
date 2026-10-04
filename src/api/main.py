"""FastAPI Server for FinBuddy AI Financial Research Assistant."""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    CreditApplicantSchema,
    CreditScoreResponse,
    DriftCheckRequest,
    HealthResponse,
)
from src.ml.dataset import FEATURE_NAMES, generate_credit_dataset
from src.ml.explain import CreditRiskExplainer
from src.ml.drift import DriftDetector
from src.agents.financial_agent import FinBuddyAgent

app = FastAPI(
    title="FinBuddy API",
    description="Backend API for AI Financial Research, Explainable Credit Risk (XGBoost + SHAP), and Drift Monitoring.",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons lazily
agent_instance = None
explainer_instance = None
drift_detector_instance = None


def get_agent():
    global agent_instance
    if agent_instance is None:
        agent_instance = FinBuddyAgent()
    return agent_instance


def get_explainer():
    global explainer_instance
    if explainer_instance is None:
        explainer_instance = CreditRiskExplainer()
    return explainer_instance


def get_drift_detector():
    global drift_detector_instance
    if drift_detector_instance is None:
        drift_detector_instance = DriftDetector()
    return drift_detector_instance


@app.get("/api/health", response_model=HealthResponse)
def health_check():
    """Returns system status and model readiness."""
    model_exists = (
        os.path.exists("xgboost_loan_model.joblib")
        or os.path.exists("xgboost_loan_model.json")
        or os.path.exists("models/xgboost_loan_model.joblib")
        or os.path.exists("models/credit_risk_xgb.pkl")
    )
    explainer_exists = os.path.exists("models/shap_explainer.pkl") or model_exists
    
    return {
        "status": "healthy" if model_exists else "needs_training",
        "version": "1.0.0",
        "model_loaded": model_exists,
        "explainer_loaded": explainer_exists,
        "features": FEATURE_NAMES,
    }


@app.post("/api/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    """Invokes the multi-tool Agentic LLM for financial inquiries."""
    agent = get_agent()
    result = agent.run(query=req.query, chat_history=req.chat_history)
    return result


@app.post("/api/credit/score", response_model=CreditScoreResponse)
def credit_score_endpoint(req: CreditApplicantSchema):
    """Underwrites a credit application with XGBoost probability and SHAP attribution."""
    try:
        explainer = get_explainer()
        applicant_dict = req.model_dump()
        if applicant_dict.get("loan_percent_income") is None:
            applicant_dict["loan_percent_income"] = round(
                applicant_dict["loan_amnt"] / max(applicant_dict["person_income"], 1.0), 3
            )
            
        result = explainer.predict_and_explain(applicant_dict)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/credit/metrics")
def get_model_metrics():
    """Fetches model training and validation metrics."""
    metrics_path = "models/model_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="Model metrics not found. Run training first.")


@app.post("/api/monitoring/drift")
def drift_monitoring_endpoint(req: DriftCheckRequest):
    """Runs PSI and KS-test drift diagnostics on simulated or incoming production traffic."""
    try:
        detector = get_drift_detector()
        # Generate production test batch with simulated macroeconomic drift
        prod_df = generate_credit_dataset(
            n_samples=req.sample_size,
            random_state=99,
            drift_factor=req.drift_factor
        )
        report = detector.evaluate_drift(prod_df)
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
