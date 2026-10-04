"""Training pipeline for XGBoost Credit Risk Classifier on Kaggle Credit Risk Dataset with SHAP."""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)
import xgboost as xgb
import shap

from src.ml.dataset import (
    load_and_preprocess_kaggle_dataset,
    save_baseline_dataset,
    FEATURE_NAMES,
    TARGET_COL,
)


def train_credit_model(
    data_path: str = "data/credit_risk_dataset.csv",
    model_output_dir: str = "models",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains an XGBoost credit risk classifier on the Kaggle dataset and saves model + SHAP artifacts.
    """
    os.makedirs(model_output_dir, exist_ok=True)
    
    print("Loading and preprocessing Kaggle credit risk dataset...")
    df_raw, X, y, meta = load_and_preprocess_kaggle_dataset(data_path)
    
    # Save the processed baseline CSV
    save_baseline_dataset(input_csv=data_path, output_csv="data/credit_baseline.csv")
    
    # Split train, val, test (70/15/15) stratified
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    
    print(f"Total: {X.shape[0]} | Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Default rate: {y_train.mean():.2%}")
    
    # Initialize and train XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=350,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.88,
        colsample_bytree=0.88,
        scale_pos_weight=1.2,
        gamma=0.05,
        min_child_weight=2,
        eval_metric=["auc", "logloss"],
        random_state=random_state,
        n_jobs=-1
    )
    
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    # Evaluate on holdout test set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.46).astype(int)
    
    roc_auc = float(roc_auc_score(y_test, y_pred_proba))
    pr_auc = float(average_precision_score(y_test, y_pred_proba))
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred))
    rec = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    brier = float(brier_score_loss(y_test, y_pred_proba))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    metrics = {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "brier_score": round(brier, 4),
        "decision_threshold": 0.46,
        "confusion_matrix": cm,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "features": FEATURE_NAMES,
    }
    
    print("\n--- Evaluation Metrics on Holdout Test Set ---")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Save Model Artifacts in multiple formats
    model_path = os.path.join(model_output_dir, "credit_risk_xgb.pkl")
    joblib.dump(model, model_path)
    
    # Save Joblib Bundle
    bundle_path = os.path.join(model_output_dir, "xgboost_loan_model.joblib")
    joblib.dump({
        "model": model,
        "optimal_threshold": 0.6903,
        "feature_names": FEATURE_NAMES
    }, bundle_path)
    
    # Save native XGBoost JSON model
    json_model_path = os.path.join(model_output_dir, "xgboost_loan_model.json")
    model.save_model(json_model_path)
    
    print(f"Models saved to {model_output_dir}: .joblib, .json, .pkl")
    
    # Fit & Save SHAP TreeExplainer
    print("Fitting SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    explainer_path = os.path.join(model_output_dir, "shap_explainer.pkl")
    joblib.dump(explainer, explainer_path)
    print(f"SHAP explainer saved to: {explainer_path}")
    
    # Save Metrics JSON
    metrics_path = os.path.join(model_output_dir, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")
    
    return metrics


if __name__ == "__main__":
    train_credit_model()
