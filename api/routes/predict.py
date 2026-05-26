from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from core.database import get_db
from models.schemas import AssetInput, RULPredictionResponse
from services.ml_service import predict_rul

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
