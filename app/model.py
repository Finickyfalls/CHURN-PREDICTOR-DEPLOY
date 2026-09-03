# Loads the trained model and preprocessing objects, and turns a raw form
# submission (human-readable values like "Month-to-month", "Fiber optic")
# into the exact numeric feature vector the model expects.

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

BINARY_MAP_COLS = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
MULTI_CAT_COLS = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]


def load_artifacts():
    model = joblib.load(os.path.join(MODELS_DIR, "churn_best_model.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "churn_scaler.joblib"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "churn_feature_columns.joblib"))
    numeric_cols = joblib.load(os.path.join(MODELS_DIR, "churn_numeric_cols.joblib"))
    return model, scaler, feature_columns, numeric_cols


def preprocess_input(raw: dict, scaler, feature_columns, numeric_cols):
    """Mirrors the exact preprocessing steps used in training (Part 1), so a raw
    form submission gets encoded and scaled identically to the training data."""

    df = pd.DataFrame([raw])

    # Binary Yes/No and gender columns -> 0/1, same mapping as training
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    for col in BINARY_MAP_COLS:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # One-hot encode multi-category columns -- identical call to training
    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True, dtype=int)

    # Align to the exact column set/order the model was trained on. Any
    # dummy column not produced by this single row (e.g. because this
    # customer's category was the "dropped" baseline) is filled with 0.
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Scale numeric features using the SAME scaler fitted during training
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df


def predict_churn(raw: dict, model, scaler, feature_columns, numeric_cols, top_k=5):
    row_df = preprocess_input(raw, scaler, feature_columns, numeric_cols)
    probability = float(model.predict_proba(row_df)[0][1])

    # Per-customer explanation: XGBoost can report exactly how much each
    # feature pushed THIS specific prediction up or down (not just a global
    # "importance" chart that looks the same for every customer).
    top_drivers = []
    try:
        booster = model.get_booster()
        dmatrix = xgb.DMatrix(row_df, feature_names=feature_columns)
        contribs = booster.predict(dmatrix, pred_contribs=True)[0][:-1]  # drop bias term
        contrib_series = pd.Series(contribs, index=feature_columns)

        # Only factors pushing risk UP are useful to show as "why this customer is at risk"
        top_positive = contrib_series[contrib_series > 0].sort_values(ascending=False).head(top_k)
        top_drivers = [{"feature": _clean_label(f), "impact": float(v)} for f, v in top_positive.items()]
    except Exception:
        # Fallback for non-XGBoost models: global feature importances instead
        if hasattr(model, "feature_importances_"):
            importances = pd.Series(model.feature_importances_, index=feature_columns)
            top = importances.sort_values(ascending=False).head(top_k)
            top_drivers = [{"feature": _clean_label(f), "impact": float(v)} for f, v in top.items()]

    if probability >= 0.6:
        risk_band = "High"
    elif probability >= 0.3:
        risk_band = "Medium"
    else:
        risk_band = "Low"

    return {
        "probability": probability,
        "risk_band": risk_band,
        "top_drivers": top_drivers,
    }


def _clean_label(feature_name):
    """Turns a raw one-hot column name like "Contract_Two year" into a
    readable label like "Contract: Two year" for display in the UI."""
    for cat_col in MULTI_CAT_COLS:
        if feature_name.startswith(cat_col + "_"):
            return f"{cat_col}: {feature_name[len(cat_col) + 1:]}"
    return feature_name
