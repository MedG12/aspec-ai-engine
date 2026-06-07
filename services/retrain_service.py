import os
import json
import shutil
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sqlalchemy import text
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from core.database import SessionLocal

logger = logging.getLogger(__name__)

BASE_MODEL_PATH = "ml_models/RUL Model/"
BACKUP_MODEL_PATH = "ml_models/RUL Model/_backup/"


def fetch_training_data() -> pd.DataFrame:
    """
    Fetch training data from the database by joining assets and maintenance_logs.
    Computes the same features used by the prediction endpoint.
    """
    db = SessionLocal()
    try:
        # Ground truth RUL = (replacement_date - instalation_date) in years
        # Only assets that have been replaced provide real observed lifespans
        query = text("""
            SELECT 
                a.asset_id,
                a.asset_type AS `Tipe`,
                a.building AS `Lokasi Gedung`,
                a.floor AS `Lokasi Lantai`,
                a.zone AS `Lokasi Zona`,
                a.instalation_date,
                a.operational_hours,
                r.replacement_date,
                COUNT(m.ticket_id) AS `Total komplain`,
                COALESCE(SUM(m.repair_cost), 0) AS `Total biaya perbaikan`
            FROM assets a
            INNER JOIN asset_replacements r ON a.asset_id = r.old_asset_id
            LEFT JOIN maintenance_logs m ON a.asset_id = m.asset_id
            WHERE a.instalation_date IS NOT NULL
              AND r.replacement_date IS NOT NULL
            GROUP BY 
                a.asset_id, a.asset_type, a.building, a.floor, a.zone,
                a.instalation_date, a.operational_hours, r.replacement_date
        """)

        results = db.execute(query).mappings().fetchall()

        rows = []
        for row in results:
            # Parse dates
            inst_date = row["instalation_date"]
            repl_date = row["replacement_date"]
            if isinstance(inst_date, str):
                inst_date = datetime.strptime(inst_date, "%Y-%m-%d").date()
            if isinstance(repl_date, str):
                repl_date = datetime.strptime(repl_date, "%Y-%m-%d").date()

            # RUL ground truth = lifespan in years (replacement_date - instalation_date)
            lifespan_days = (repl_date - inst_date).days
            if lifespan_days <= 0:
                continue  # Skip invalid data
            rul_years = lifespan_days / 365.25

            # Compute Operating_Hours relative to replacement_date (state at end-of-life)
            operating_hours = 0.0
            if row["operational_hours"]:
                delta_days = max(0, lifespan_days)
                operating_hours = delta_days * (5.0 / 7.0) * float(row["operational_hours"])

            rows.append({
                "Total komplain": int(row["Total komplain"]),
                "Total biaya perbaikan": float(row["Total biaya perbaikan"]),
                "Lokasi Lantai": row["Lokasi Lantai"] or 1,
                "Operating_Hours": round(operating_hours, 2),
                "Tipe": row["Tipe"] or "Unknown",
                "Lokasi Gedung": row["Lokasi Gedung"] or "Unknown",
                "Lokasi Zona": row["Lokasi Zona"] or "Unknown",
                "RUL": round(rul_years, 4),
            })

        return pd.DataFrame(rows)
    finally:
        db.close()


def retrain_xgboost_model() -> dict:
    """
    Retrain the XGBoost RUL model using fresh data from the database.
    Returns a dict with training metrics.
    """
    logger.info("=== Starting XGBoost Model Retrain ===")

    # --- 1. Fetch data ---
    df = fetch_training_data()
    if len(df) < 20:
        msg = f"Not enough training data ({len(df)} rows). Skipping retrain."
        logger.warning(msg)
        return {"success": False, "message": msg}

    logger.info(f"Fetched {len(df)} training samples from database.")

    # --- 2. Load existing metrics for hyperparameters ---
    metrics_path = os.path.join(BASE_MODEL_PATH, "metrics.json")
    with open(metrics_path, "r") as f:
        existing_metrics = json.load(f)

    hyperparams = existing_metrics["hyperparameters"]["XGBoost"]

    # --- 3. Prepare features and target ---
    target = "RUL"
    feature_order = existing_metrics["feature_order"]
    cat_features = existing_metrics["categorical_features"]

    X = df[feature_order].copy()
    y = df[target].values.reshape(-1, 1)

    # --- 4. Fit encoders and scalers ---
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat_features] = encoder.fit_transform(X[cat_features])

    feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(X)

    target_scaler = StandardScaler()
    y_scaled = target_scaler.fit_transform(y).ravel()

    # --- 5. Train/test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    # --- 6. Train XGBoost ---
    model = XGBRegressor(
        n_estimators=hyperparams.get("n_estimators", 400),
        learning_rate=hyperparams.get("learning_rate", 0.05),
        max_depth=hyperparams.get("max_depth", 5),
        subsample=hyperparams.get("subsample", 0.8),
        colsample_bytree=hyperparams.get("colsample_bytree", 0.8),
        min_child_weight=hyperparams.get("min_child_weight", 3),
        gamma=hyperparams.get("gamma", 0.0),
        reg_alpha=hyperparams.get("reg_alpha", 0.1),
        reg_lambda=hyperparams.get("reg_lambda", 1.0),
        objective=hyperparams.get("objective", "reg:squarederror"),
        random_state=hyperparams.get("random_state", 42),
        n_jobs=hyperparams.get("n_jobs", -1),
    )

    model.fit(X_train, y_train)

    # --- 7. Evaluate ---
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_pred_scaled = model.predict(X_test)
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_actual, y_pred)))
    mae = float(mean_absolute_error(y_actual, y_pred))
    r2 = float(r2_score(y_actual, y_pred))

    logger.info(f"Retrain metrics — RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

    # --- 8. Backup existing models ---
    os.makedirs(BACKUP_MODEL_PATH, exist_ok=True)
    files_to_backup = [
        "xgboost_model.pkl",
        "ordinal_encoder.pkl",
        "feature_scaler.pkl",
        "target_scaler.pkl",
        "metrics.json",
    ]
    for fname in files_to_backup:
        src = os.path.join(BASE_MODEL_PATH, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(BACKUP_MODEL_PATH, fname))

    logger.info("Existing models backed up.")

    # --- 9. Save new models ---
    try:
        joblib.dump(model, os.path.join(BASE_MODEL_PATH, "xgboost_model.pkl"))
        joblib.dump(encoder, os.path.join(BASE_MODEL_PATH, "ordinal_encoder.pkl"))
        joblib.dump(feature_scaler, os.path.join(BASE_MODEL_PATH, "feature_scaler.pkl"))
        joblib.dump(target_scaler, os.path.join(BASE_MODEL_PATH, "target_scaler.pkl"))

        # Update metrics
        existing_metrics["metrics"]["XGBoost"] = {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4),
            "R2": round(r2, 4),
        }
        existing_metrics["n_train"] = len(X_train)
        existing_metrics["n_test"] = len(X_test)
        existing_metrics["last_retrained_at"] = datetime.now().isoformat()

        with open(metrics_path, "w") as f:
            json.dump(existing_metrics, f, indent=2)

        logger.info("New models saved successfully.")
    except Exception as e:
        # Rollback: restore backup
        logger.error(f"Error saving models, rolling back: {e}")
        for fname in files_to_backup:
            backup_src = os.path.join(BACKUP_MODEL_PATH, fname)
            if os.path.exists(backup_src):
                shutil.copy2(backup_src, os.path.join(BASE_MODEL_PATH, fname))
        raise

    # --- 10. Force reload models in ml_service ---
    from services.ml_service import reload_xgboost_models
    reload_xgboost_models()

    result = {
        "success": True,
        "message": "XGBoost model retrained successfully.",
        "samples": len(df),
        "metrics": {"RMSE": rmse, "MAE": mae, "R2": r2},
        "retrained_at": datetime.now().isoformat(),
    }

    logger.info(f"=== Retrain Complete: {result} ===")
    return result
