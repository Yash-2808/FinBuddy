"""MLOps Data & Concept Drift Monitoring Engine."""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from scipy.stats import ks_2samp

from src.ml.dataset import FEATURE_NAMES, generate_credit_dataset


def calculate_psi(
    expected: np.ndarray,
    actual: np.ndarray,
    num_buckets: int = 10
) -> float:
    """
    Calculates the Population Stability Index (PSI) between baseline (expected)
    and current (actual) feature distributions.
    """
    # Remove NaN values
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantiles on expected distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bucket_bounds = np.percentile(expected, percentiles)
    # Ensure unique bucket edges
    bucket_bounds = np.unique(bucket_bounds)
    
    if len(bucket_bounds) < 2:
        return 0.0

    # Set outer edges to -inf and +inf
    bucket_bounds[0] = -np.inf
    bucket_bounds[-1] = np.inf

    # Count frequencies
    exp_counts, _ = np.histogram(expected, bins=bucket_bounds)
    act_counts, _ = np.histogram(actual, bins=bucket_bounds)

    # Convert to proportions with epsilon smoothing
    eps = 1e-4
    exp_prop = np.maximum(exp_counts / len(expected), eps)
    act_prop = np.maximum(act_counts / len(actual), eps)

    # Re-normalize
    exp_prop = exp_prop / np.sum(exp_prop)
    act_prop = act_prop / np.sum(act_prop)

    # Compute PSI
    psi_values = (act_prop - exp_prop) * np.log(act_prop / exp_prop)
    return float(np.sum(psi_values))


class DriftDetector:
    """Monitors incoming production features against training baseline for data drift."""

    def __init__(self, baseline_data_path: str = "data/credit_baseline.csv"):
        self.baseline_data_path = baseline_data_path
        self.baseline_df: Optional[pd.DataFrame] = None
        self._load_baseline()

    def _load_baseline(self) -> None:
        if os.path.exists(self.baseline_data_path):
            self.baseline_df = pd.read_csv(self.baseline_data_path)
        else:
            print(f"Generating baseline for drift monitoring at {self.baseline_data_path}...")
            self.baseline_df = generate_credit_dataset(n_samples=10000, random_state=42)
            os.makedirs(os.path.dirname(self.baseline_data_path) or ".", exist_ok=True)
            self.baseline_df.to_csv(self.baseline_data_path, index=False)

    def evaluate_drift(
        self,
        production_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Runs PSI and Kolmogorov-Smirnov statistical tests across all features.
        """
        report_features = []
        drift_detected_count = 0
        moderate_drift_count = 0

        for col in FEATURE_NAMES:
            if col not in production_df.columns:
                continue

            base_series = self.baseline_df[col].values
            prod_series = production_df[col].values

            # Calculate PSI
            psi_score = calculate_psi(base_series, prod_series)
            
            # Calculate Kolmogorov-Smirnov Test
            ks_stat, p_val = ks_2samp(base_series, prod_series)

            # Drift Severity Classification
            if psi_score >= 0.25 or (p_val < 0.001 and ks_stat > 0.15):
                status = "SIGNIFICANT DRIFT (Action Required)"
                severity = "HIGH"
                drift_detected_count += 1
            elif psi_score >= 0.10 or p_val < 0.01:
                status = "MODERATE DRIFT (Monitor)"
                severity = "MEDIUM"
                moderate_drift_count += 1
            else:
                status = "STABLE"
                severity = "LOW"

            report_features.append({
                "feature": col,
                "psi_score": round(psi_score, 4),
                "ks_statistic": round(float(ks_stat), 4),
                "p_value": float(f"{p_val:.4e}"),
                "status": status,
                "severity": severity,
                "baseline_mean": round(float(np.mean(base_series)), 2),
                "production_mean": round(float(np.mean(prod_series)), 2),
                "baseline_std": round(float(np.std(base_series)), 2),
                "production_std": round(float(np.std(prod_series)), 2),
            })

        # Overall Status
        if drift_detected_count > 0:
            overall_verdict = "ALERT: Critical feature drift detected. Model retraining or data pipeline inspection recommended."
            health_score = max(0, 100 - (drift_detected_count * 25 + moderate_drift_count * 10))
        elif moderate_drift_count > 0:
            overall_verdict = "WARNING: Moderate distribution shift observed in one or more features."
            health_score = max(50, 100 - (moderate_drift_count * 10))
        else:
            overall_verdict = "HEALTHY: All features conform to training distribution baseline."
            health_score = 100

        return {
            "overall_verdict": overall_verdict,
            "system_health_score": health_score,
            "features_analyzed": len(report_features),
            "high_drift_features": drift_detected_count,
            "moderate_drift_features": moderate_drift_count,
            "feature_metrics": report_features,
            "baseline_samples": len(self.baseline_df),
            "production_samples": len(production_df),
        }
