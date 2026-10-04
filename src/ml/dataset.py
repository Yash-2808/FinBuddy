"""Dataset loader and preprocessor for Kaggle Credit Risk Dataset.
Handles categorical encodings, missing value imputation, and feature alignment.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List, Optional


TARGET_COL = "loan_status"

RAW_NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_OPTIONS = {
    "person_home_ownership": ["RENT", "OWN", "MORTGAGE", "OTHER"],
    "loan_intent": ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
    "loan_grade": ["A", "B", "C", "D", "E", "F", "G"],
    "cb_person_default_on_file": ["N", "Y"],
}

# The complete 22-feature encoded list used for XGBoost training and SHAP
FEATURE_NAMES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "person_home_ownership_OTHER",
    "person_home_ownership_OWN",
    "person_home_ownership_RENT",
    "loan_intent_EDUCATION",
    "loan_intent_HOMEIMPROVEMENT",
    "loan_intent_MEDICAL",
    "loan_intent_PERSONAL",
    "loan_intent_VENTURE",
    "loan_grade_B",
    "loan_grade_C",
    "loan_grade_D",
    "loan_grade_E",
    "loan_grade_F",
    "loan_grade_G",
    "cb_person_default_on_file_Y",
]


def load_and_preprocess_kaggle_dataset(
    csv_path: str = "data/credit_risk_dataset.csv"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Loads and preprocesses the Kaggle Credit Risk Dataset matching the new XGBoost model pipeline.
    Cleans outliers, drops nulls, and one-hot encodes categoricals (drop_first=True).
    
    Returns:
        df_clean: Full cleaned DataFrame (with raw categoricals)
        X: Encoded feature matrix (N x 22)
        y: Binary target series (loan_status)
        meta: Preprocessing metadata (medians, categorical levels)
    """
    if not os.path.isabs(csv_path):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        resolved_path = os.path.join(base_dir, csv_path)
        if os.path.exists(resolved_path):
            csv_path = resolved_path

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Kaggle credit dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)

    # 1. Drop NaNs and filter extreme outliers (age <= 90)
    df = df.dropna().copy()
    df = df[df["person_age"] <= 90].copy()

    # 2. Winsorize income at 99th percentile
    income_cap = float(df["person_income"].quantile(0.99))
    df["person_income"] = df["person_income"].clip(upper=income_cap)

    # 3. Categorical encoding with drop_first=True
    cat_cols = list(CATEGORICAL_OPTIONS.keys())
    df_encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Ensure all expected columns exist
    for col in FEATURE_NAMES:
        if col not in df_encoded.columns:
            df_encoded[col] = 0.0

    X = df_encoded[FEATURE_NAMES].astype(float)
    y = df_encoded[TARGET_COL].astype(int)

    meta = {
        "income_cap": income_cap,
        "categorical_options": CATEGORICAL_OPTIONS,
        "feature_names": FEATURE_NAMES,
        "n_samples": len(df),
    }

    return df, X, y, meta


def transform_applicant_input(
    applicant_data: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Transforms a single applicant dictionary into a 1-row DataFrame aligned with FEATURE_NAMES (22 features).
    Supports both raw categoricals and numeric inputs.
    """
    row = {}

    # Extract numeric values with robust fallbacks
    age = float(applicant_data.get("person_age", 30))
    inc = float(applicant_data.get("person_income", 60000))
    emp = float(applicant_data.get("person_emp_length", 4.0))
    loan = float(applicant_data.get("loan_amnt", 12000))
    int_rate = float(applicant_data.get("loan_int_rate", 11.0))
    
    # Calculate loan percent of income if not explicitly given
    lti = float(applicant_data.get("loan_percent_income", loan / max(inc, 1.0)))
    cred_hist = float(applicant_data.get("cb_person_cred_hist_length", 5.0))

    row["person_age"] = age
    row["person_income"] = inc
    row["person_emp_length"] = emp
    row["loan_amnt"] = loan
    row["loan_int_rate"] = int_rate
    row["loan_percent_income"] = round(lti, 3)
    row["cb_person_cred_hist_length"] = cred_hist

    # Extract categoricals (drop_first=True reference categories: MORTGAGE, DEBTCONSOLIDATION, A, N)
    home_ownership = str(applicant_data.get("person_home_ownership", "RENT")).upper()
    intent = str(applicant_data.get("loan_intent", "PERSONAL")).upper()
    grade = str(applicant_data.get("loan_grade", "B")).upper()
    default_file = str(applicant_data.get("cb_person_default_on_file", "N")).upper()

    row["person_home_ownership_OTHER"] = 1.0 if home_ownership == "OTHER" else 0.0
    row["person_home_ownership_OWN"] = 1.0 if home_ownership == "OWN" else 0.0
    row["person_home_ownership_RENT"] = 1.0 if home_ownership == "RENT" else 0.0

    row["loan_intent_EDUCATION"] = 1.0 if intent == "EDUCATION" else 0.0
    row["loan_intent_HOMEIMPROVEMENT"] = 1.0 if intent == "HOMEIMPROVEMENT" else 0.0
    row["loan_intent_MEDICAL"] = 1.0 if intent == "MEDICAL" else 0.0
    row["loan_intent_PERSONAL"] = 1.0 if intent == "PERSONAL" else 0.0
    row["loan_intent_VENTURE"] = 1.0 if intent == "VENTURE" else 0.0

    row["loan_grade_B"] = 1.0 if grade == "B" else 0.0
    row["loan_grade_C"] = 1.0 if grade == "C" else 0.0
    row["loan_grade_D"] = 1.0 if grade == "D" else 0.0
    row["loan_grade_E"] = 1.0 if grade == "E" else 0.0
    row["loan_grade_F"] = 1.0 if grade == "F" else 0.0
    row["loan_grade_G"] = 1.0 if grade == "G" else 0.0

    row["cb_person_default_on_file_Y"] = 1.0 if default_file in ["Y", "YES", "1", 1] else 0.0

    df_out = pd.DataFrame([row])[FEATURE_NAMES]
    return df_out


def save_baseline_dataset(
    input_csv: str = "data/credit_risk_dataset.csv",
    output_csv: str = "data/credit_baseline.csv"
) -> str:
    """Processes the Kaggle dataset and writes the standardized baseline CSV."""
    df_raw, X, y, meta = load_and_preprocess_kaggle_dataset(input_csv)
    df_out = X.copy()
    df_out[TARGET_COL] = y
    
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    
    meta_path = os.path.join(os.path.dirname(output_csv) or ".", "preprocessor_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        
    return output_csv


def generate_credit_dataset(
    n_samples: int = 1000,
    random_state: int = 42,
    drift_factor: float = 0.0
) -> pd.DataFrame:
    """
    Generates synthetic or drifted production batches sampled from the Kaggle dataset distribution
    for the MLOps Drift Radar (Tab 4).
    """
    np.random.seed(random_state)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    baseline_path = os.path.join(base_dir, "data", "credit_baseline.csv")
    
    if os.path.exists(baseline_path):
        base_df = pd.read_csv(baseline_path)
    else:
        _, base_df, _, _ = load_and_preprocess_kaggle_dataset()

    sample_df = base_df.sample(n=min(n_samples, len(base_df)), replace=True, random_state=random_state).copy()
    
    if drift_factor > 0:
        # Apply macroeconomic drift shifts
        sample_df["loan_int_rate"] = (sample_df["loan_int_rate"] * (1.0 + drift_factor * 0.35)).clip(4.0, 35.0)
        sample_df["person_income"] = (sample_df["person_income"] * (1.0 - drift_factor * 0.20)).clip(4000, 500000)
        sample_df["loan_percent_income"] = (sample_df["loan_percent_income"] * (1.0 + drift_factor * 0.30)).clip(0.01, 1.0)
        sample_df["loan_amnt"] = (sample_df["loan_amnt"] * (1.0 + drift_factor * 0.15)).clip(500, 45000)

    return sample_df


if __name__ == "__main__":
    out = save_baseline_dataset()
    print(f"Kaggle baseline processed and saved to: {out}")
