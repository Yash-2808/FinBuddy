"""Unit and integration tests for ML model, SHAP explainability, and drift detection."""

import os
import unittest
import tempfile
import pandas as pd
import numpy as np

from src.ml.dataset import generate_credit_dataset, load_and_preprocess_kaggle_dataset, FEATURE_NAMES, TARGET_COL
from src.ml.train import train_credit_model
from src.ml.explain import CreditRiskExplainer
from src.ml.drift import DriftDetector, calculate_psi


class TestCreditMLPipeline(unittest.TestCase):

    def test_dataset_generation(self):
        """Verify Kaggle baseline dataset generator shapes, types, and value constraints."""
        df = generate_credit_dataset(n_samples=500, random_state=42)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 500)
        for col in FEATURE_NAMES:
            self.assertIn(col, df.columns)
        self.assertIn(TARGET_COL, df.columns)
        self.assertGreaterEqual(df["person_age"].min(), 18)
        self.assertTrue(set(df[TARGET_COL].unique()).issubset({0, 1}))

    def test_model_training_and_artifacts(self):
        """Verify XGBoost training pipeline and metrics output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = os.path.join(tmp_dir, "models")
            
            metrics = train_credit_model(
                data_path="data/credit_risk_dataset.csv",
                model_output_dir=output_dir,
                random_state=42
            )
            
            self.assertIn("roc_auc", metrics)
            self.assertGreater(metrics["roc_auc"], 0.70)
            self.assertTrue(os.path.exists(os.path.join(output_dir, "credit_risk_xgb.pkl")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "shap_explainer.pkl")))
            self.assertTrue(os.path.exists(os.path.join(output_dir, "model_metrics.json")))

    def test_shap_explainer_inference(self):
        """Verify SHAP local attribution logic and output structure."""
        explainer = CreditRiskExplainer()
        
        sample_applicant = {
            "person_age": 30,
            "person_income": 80000.0,
            "person_emp_length": 6.0,
            "loan_amnt": 10000.0,
            "loan_int_rate": 8.5,
            "loan_percent_income": 0.125,
            "cb_person_cred_hist_length": 8.0,
            "person_home_ownership": "MORTGAGE",
            "loan_intent": "EDUCATION",
            "loan_grade": "A",
            "cb_person_default_on_file": "N",
        }
        
        result = explainer.predict_and_explain(sample_applicant)
        self.assertIn("default_probability", result)
        self.assertGreaterEqual(result["default_probability"], 0.0)
        self.assertLessEqual(result["default_probability"], 1.0)
        self.assertIn("recommendation", result)
        self.assertGreater(len(result["feature_contributions"]), 0)
        self.assertLessEqual(len(result["top_risk_drivers"]), 3)
        self.assertLessEqual(len(result["top_protective_factors"]), 3)

    def test_drift_detection_psi(self):
        """Verify PSI calculation and DriftDetector under baseline vs drifted distributions."""
        np.random.seed(42)
        baseline = np.random.normal(100, 15, 2000)
        same_dist = np.random.normal(100, 15, 2000)
        drifted_dist = np.random.normal(140, 25, 2000)
        
        psi_stable = calculate_psi(baseline, same_dist)
        psi_drifted = calculate_psi(baseline, drifted_dist)
        
        self.assertLess(psi_stable, 0.10)  # Stable
        self.assertGreater(psi_drifted, 0.25)  # Significant drift detected


if __name__ == "__main__":
    unittest.main()
