import pandas as pd
import joblib
import json
import numpy as np
import os
from models.schemas import AssetInput, XGBoostRULRequest

# Base path for RUL models
BASE_MODEL_PATH = "ml_models/RUL Model/"

# Global variables for models to avoid reloading on every request
xgboost_model = None
xgboost_metrics = None
xgboost_encoder = None
xgboost_feature_scaler = None
xgboost_target_scaler = None

def load_xgboost_models():
    global xgboost_model, xgboost_metrics, xgboost_encoder, xgboost_feature_scaler, xgboost_target_scaler
    
    # Check if any required component is None
    if any(v is None for v in [xgboost_model, xgboost_metrics, xgboost_encoder, xgboost_feature_scaler, xgboost_target_scaler]):
        try:
            with open(os.path.join(BASE_MODEL_PATH, "metrics.json"), "r") as f:
                xgboost_metrics = json.load(f)
            
            xgboost_encoder = joblib.load(os.path.join(BASE_MODEL_PATH, "ordinal_encoder.pkl"))
            xgboost_feature_scaler = joblib.load(os.path.join(BASE_MODEL_PATH, "feature_scaler.pkl"))
            xgboost_target_scaler = joblib.load(os.path.join(BASE_MODEL_PATH, "target_scaler.pkl"))
            xgboost_model = joblib.load(os.path.join(BASE_MODEL_PATH, "xgboost_model.pkl"))
        except Exception as e:
            print(f"Error loading XGBoost models: {e}")
            raise RuntimeError(f"Failed to load XGBoost models: {e}")

def predict_rul(asset_input: AssetInput) -> float:
    """
    Logic to load .pkl/.h5 models and run predictions using Pandas/Sklearn/Keras.
    """
    # Dummy prediction algorithm for placeholder
    dummy_rul = asset_input.operating_hours * 0.5 + 100.0
    return float(dummy_rul)

def predict_xgboost_rul(request: XGBoostRULRequest) -> float:
    load_xgboost_models()
    
    # Explicitly check for metrics and models to provide a clearer error and satisfy type checkers
    if xgboost_metrics is None or xgboost_encoder is None or xgboost_feature_scaler is None or xgboost_model is None:
        raise RuntimeError("XGBoost models or metrics are missing. Model loading might have failed.")

    # Convert Pydantic model to DataFrame using alias names
    data = request.model_dump(by_alias=True)
    # Remove asset_id before processing
    if "asset_id" in data:
        del data["asset_id"]

    df = pd.DataFrame([data])

    # Ensure correct feature order
    feature_order = xgboost_metrics["feature_order"]
    df = df[feature_order]

    # Encoding categorical features
    cat_features = xgboost_metrics["categorical_features"]
    df[cat_features] = xgboost_encoder.transform(df[cat_features])

    # Scaling features
    df_scaled = xgboost_feature_scaler.transform(df)

    # Predict
    pred_scaled = xgboost_model.predict(df_scaled)

    # Inverse transform target
    if hasattr(xgboost_target_scaler, "inverse_transform"):
        if len(pred_scaled.shape) == 1:
            pred_scaled = pred_scaled.reshape(-1, 1)
        pred_actual = xgboost_target_scaler.inverse_transform(pred_scaled)
        rul_value = float(pred_actual[0][0])
    else:
        rul_value = float(pred_scaled[0])
        
    return max(0.0, rul_value) # Prevent negative RUL

def reload_xgboost_models():
    """Reset cached models to force a fresh load on next prediction (e.g. after retrain)."""
    global xgboost_model, xgboost_metrics, xgboost_encoder, xgboost_feature_scaler, xgboost_target_scaler
    xgboost_model = None
    xgboost_metrics = None
    xgboost_encoder = None
    xgboost_feature_scaler = None
    xgboost_target_scaler = None
    load_xgboost_models()
