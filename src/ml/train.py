"""Training pipeline for XGBoost Credit Risk Classifier with SHAP Explainer serialization."""

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

from src.ml.dataset import generate_credit_dataset, FEATURE_NAMES, TARGET_COL, save_baseline_dataset


def train_credit_model(
    data_path: str = "data/credit_baseline.csv",
    model_output_dir: str = "models",
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains an XGBoost credit risk classifier and saves the model + SHAP artifacts.
    """
    os.makedirs(model_output_dir, exist_ok=True)
    
    # Always regenerate fresh baseline dataset with the new refined generator
    print("Generating fresh, high-accuracy baseline dataset...")
    df = generate_credit_dataset(n_samples=25000, random_state=random_state)
    os.makedirs(os.path.dirname(data_path) or ".", exist_ok=True)
    df.to_csv(data_path, index=False)
        
    X = df[FEATURE_NAMES]
    y = df[TARGET_COL]
    
    # Split train, val, test (70/15/15)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )
    
    print(f"Training set: {X_train.shape[0]} samples | Validation: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    print(f"Default rate in training: {y_train.mean():.2%}")
    
    # Initialize and train XGBoost Classifier
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=2,
        gamma=0.05,
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
    
    # Evaluate on Test Set
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
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
        "confusion_matrix": cm,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "features": FEATURE_NAMES
    }
    
    print("\n--- Model Performance Metrics ---")
    for k, v in metrics.items():
        if k != "confusion_matrix" and k != "features":
            print(f"{k.upper()}: {v}")
            
    # Train SHAP TreeExplainer
    print("\nFitting SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    
    # Save Model Artifacts
    model_path = os.path.join(model_output_dir, "credit_risk_xgb.pkl")
    explainer_path = os.path.join(model_output_dir, "shap_explainer.pkl")
    metrics_path = os.path.join(model_output_dir, "model_metrics.json")
    
    joblib.dump(model, model_path)
    joblib.dump(explainer, explainer_path)
    
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
        
    print(f"\nArtifacts successfully saved:")
    print(f" - Model: {model_path}")
    print(f" - Explainer: {explainer_path}")
    print(f" - Metrics: {metrics_path}")
    
    return metrics


if __name__ == "__main__":
    train_credit_model()
