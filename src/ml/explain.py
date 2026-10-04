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
        model_path: str = "xgboost_loan_model.joblib",
        explainer_path: str = "models/shap_explainer.pkl"
    ):
        self.model_path = self._resolve_model_path(model_path)
        self.explainer_path = self._resolve_path(explainer_path)
        self.model: Optional[xgb.XGBClassifier] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self.optimal_threshold: float = 0.6903
        self.expected_features: List[str] = FEATURE_NAMES
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

    def _resolve_model_path(self, path: str) -> str:
        # Candidate model locations in priority order
        candidates = [
            path,
            self._resolve_path(path),
            self._resolve_path("xgboost_loan_model.joblib"),
            self._resolve_path("models/xgboost_loan_model.joblib"),
            self._resolve_path("xgboost_loan_model.json"),
            self._resolve_path("models/xgboost_loan_model.json"),
            self._resolve_path("models/credit_risk_xgb.pkl"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return self._resolve_path(path)

    def _load_artifacts(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Please check xgboost_loan_model.joblib or run training."
            )

        if self.model_path.endswith(".joblib") or self.model_path.endswith(".pkl"):
            loaded_obj = joblib.load(self.model_path)
            if isinstance(loaded_obj, dict):
                self.model = loaded_obj.get("model")
                self.optimal_threshold = float(loaded_obj.get("optimal_threshold", 0.6903))
                if "feature_names" in loaded_obj:
                    self.expected_features = loaded_obj["feature_names"]
            else:
                self.model = loaded_obj
        elif self.model_path.endswith(".json"):
            self.model = xgb.XGBClassifier()
            self.model.load_model(self.model_path)

        if os.path.exists(self.explainer_path):
            try:
                self.explainer = joblib.load(self.explainer_path)
            except Exception:
                if self.model is not None:
                    self.explainer = shap.TreeExplainer(self.model)
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
        # Automatically transform raw dictionary into 22-feature encoded DataFrame
        df_row = transform_applicant_input(applicant_data)
        
        # Predict probability of default
        prob_default = float(self.model.predict_proba(df_row)[0, 1])
        
        # Risk Category classification using optimal threshold (0.6903)
        opt_thresh = getattr(self, "optimal_threshold", 0.6903)
        if prob_default < 0.20:
            risk_tier = "LOW RISK (Prime)"
            recommendation = "APPROVE"
        elif prob_default < opt_thresh:
            risk_tier = "MODERATE RISK (Near-Prime)"
            recommendation = "APPROVE (Standard Terms)"
        elif prob_default < 0.85:
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
            "optimal_threshold": round(opt_thresh, 4),
            "base_value": round(base_val, 4),
            "narrative_explanation": narrative,
            "feature_contributions": contributions,
            "top_risk_drivers": top_risk_drivers,
            "top_protective_factors": top_protective_factors,
        }
