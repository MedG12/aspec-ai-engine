from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, List
from datetime import date

class AssetInput(BaseModel):
    tipe: str = Field(..., alias="Tipe")
    lokasi_gedung: str = Field(..., alias="Lokasi Gedung")
    lokasi_lantai: int = Field(..., alias="Lokasi Lantai")
    lokasi_zona: str = Field(..., alias="Lokasi Zona")
    operating_hours: float = Field(..., alias="Operating_Hours")
    total_komplain: int = Field(..., alias="Total komplain")
    total_biaya_perbaikan: float = Field(..., alias="Total biaya perbaikan")
    model_config = ConfigDict(populate_by_name=True)
    # class Config:
    #     # Allows accessing fields using their python names or their aliases
    #     allow_population_by_field_name = True

class XGBoostRULRequest(AssetInput):
    asset_id: int
        
class RULPredictionResponse(BaseModel):
    asset_input: AssetInput
    predicted_rul: float
    
class ChatRequest(BaseModel):
    message: str

class InsertResponse(BaseModel):
    success: bool
    mode: str
    indexed: int
    total_in_db: int
    message: str

class MaintenanceLogItem(BaseModel):
    ticket_id: int
    asset_id: int
    asset_type: str
    issue_type: Optional[str] = None
    root_cause: Optional[str] = None
    spare_parts_used: Optional[str] = None
    repair_cost: Optional[int] = None

class BatchInsertRequest(BaseModel):
    records: List[MaintenanceLogItem]
