"""Explainability engine for Credit Risk predictions using SHAP (SHapley Additive exPlanations)."""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
import xgboost as xgb
import shap

from src.ml.dataset import FEATURE_NAMES, transform_applicant_input


class CreditRiskExplainer:
    """Provides predictions and explainable SHAP breakdowns for credit underwriting."""

    def __init__(
        self,
        model_path: str = "models/credit_risk_xgb.pkl",
        explainer_path: str = "models/shap_explainer.pkl"
    ):
        self.model_path = self._resolve_path(model_path)
        self.explainer_path = self._resolve_path(explainer_path)
        self.model: Optional[xgb.XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self._load_artifacts()

    @staticmethod
    def _resolve_path(path: str) -> str:
        if os.path.exists(path):
            return path
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        alt_path = os.path.join(root_dir, path)
        if os.path.exists(alt_path):
            return alt_path
        return path

    def _load_artifacts(self) -> None:
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Please run `python -m src.ml.train` first."
            )
            
        if os.path.exists(self.explainer_path):
            self.explainer = joblib.load(self.explainer_path)
        elif self.model is not None:
            self.explainer = shap.TreeExplainer(self.model)

    def predict_and_explain(
        self,
        applicant_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculates credit default probability and local SHAP feature attribution for a single applicant.
        
        Args:
            applicant_data: Dictionary of applicant features with raw or encoded attributes.
        """
        # Automatically transform raw dictionary into 26-feature encoded DataFrame
        df_row = transform_applicant_input(applicant_data)
        
        # Predict probability of default
        prob_default = float(self.model.predict_proba(df_row)[0, 1])
        
        # Risk Category classification
        if prob_default < 0.20:
            risk_tier = "LOW RISK (Prime)"
            recommendation = "APPROVE"
        elif prob_default < 0.46:
            risk_tier = "MODERATE RISK (Near-Prime)"
            recommendation = "MANUAL REVIEW / CONDITIONAL APPROVAL"
        elif prob_default < 0.70:
            risk_tier = "HIGH RISK (Subprime)"
            recommendation = "REJECT OR REQUIRE COLLATERAL / CO-SIGNER"
        else:
            risk_tier = "VERY HIGH RISK (Critical)"
            recommendation = "REJECT"

        # Compute SHAP values
        shap_values = self.explainer(df_row)
        
        # Handle SHAP output dimensions
        if len(shap_values.values.shape) == 3:
            vals = shap_values.values[0, :, 1]
            base_val = float(shap_values.base_values[0, 1])
        else:
            vals = shap_values.values[0]
            base_val = float(shap_values.base_values[0])
            
        contributions = []
        for feat_name, shap_val in zip(FEATURE_NAMES, vals):
            feat_val = float(df_row[feat_name].iloc[0])
            # Only include non-zero one-hot indicators or numeric features
            if feat_val != 0.0 or not any(feat_name.startswith(p) for p in ["person_home_ownership_", "loan_intent_", "loan_grade_", "cb_person_default_on_file_"]):
                contributions.append({
                    "feature": feat_name,
                    "value": feat_val,
                    "shap_value": round(float(shap_val), 4),
                    "impact": "INCREASES RISK" if shap_val > 0 else "REDUCES RISK",
                    "abs_importance": round(float(abs(shap_val)), 4)
                })
            
        # Sort by absolute SHAP impact
        contributions.sort(key=lambda x: x["abs_importance"], reverse=True)
        
        top_risk_drivers = [c for c in contributions if c["shap_value"] > 0][:3]
        top_protective_factors = [c for c in contributions if c["shap_value"] < 0][:3]
        
        # Natural language synthesis
        driver_desc = ", ".join([f"{d['feature']} ({d['value']})" for d in top_risk_drivers]) or "None"
        protective_desc = ", ".join([f"{p['feature']} ({p['value']})" for p in top_protective_factors]) or "None"
        
        narrative = (
            f"Applicant default probability is {prob_default:.1%} ({risk_tier}). "
            f"Underwriting Decision: {recommendation}. "
            f"Key risk-increasing drivers: {driver_desc}. "
            f"Key protective factors: {protective_desc}."
        )

        return {
            "default_probability": round(prob_default, 4),
            "approval_score": round((1.0 - prob_default) * 100, 1),
            "risk_tier": risk_tier,
            "recommendation": recommendation,
            "base_value": round(base_val, 4),
            "narrative_explanation": narrative,
            "feature_contributions": contributions,
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_protective_factors,
        }
