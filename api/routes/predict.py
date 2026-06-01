from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from datetime import datetime
from core.database import get_db
from models.schemas import AssetInput, RULPredictionResponse, XGBoostRULRequest
from services.ml_service import predict_rul, predict_xgboost_rul

router = APIRouter()

@router.post("/predict-rul", response_model=RULPredictionResponse)
def predict_rul_endpoint(asset_input: AssetInput, db: Session = Depends(get_db)):
    """
    Predict Remaining Useful Life (RUL) for an asset.
    """
    # Call ML service to get prediction
    predicted_rul = predict_rul(asset_input)
    
    return RULPredictionResponse(
        asset_input=asset_input,
        predicted_rul=predicted_rul
    )

@router.post("/predict-xgboost-rul")
def predict_xgboost_rul_endpoint(request: XGBoostRULRequest, db: Session = Depends(get_db)):
    """
    Predict RUL using XGBoost model and update the database directly.
    """
    try:
        # 1. Predict RUL
        predicted_rul = predict_xgboost_rul(request)
        
        # 2. Check if asset exists
        asset_exists = db.execute(
            text("SELECT 1 FROM assets WHERE asset_id = :asset_id"), 
            {"asset_id": request.asset_id}
        ).scalar()
        
        if not asset_exists:
            raise HTTPException(status_code=404, detail=f"Asset with ID {request.asset_id} not found.")

        # 3. Update database
        update_stmt = text("""
            UPDATE assets 
            SET predicted_rul = :rul, last_ml_updated_at = :now 
            WHERE asset_id = :asset_id
        """)
        
        db.execute(update_stmt, {
            "rul": predicted_rul, 
            "now": datetime.now(), 
            "asset_id": request.asset_id
        })
        db.commit()

        return {
            "success": True,
            "asset_id": request.asset_id,
            "predicted_rul": predicted_rul,
            "message": "Successfully predicted and updated database."
        }
    except HTTPException as he:
        db.rollback()
        raise he
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
