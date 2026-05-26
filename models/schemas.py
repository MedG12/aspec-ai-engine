from pydantic import BaseModel, Field

class AssetInput(BaseModel):
    tipe: str = Field(..., alias="Tipe")
    lokasi_gedung: str = Field(..., alias="Lokasi Gedung")
    lokasi_lantai: str = Field(..., alias="Lokasi Lantai")
    lokasi_zona: str = Field(..., alias="Lokasi Zona")
    biaya_penggantian: float = Field(..., alias="Biaya Penggantian")
    operating_hours: float = Field(..., alias="Operating_Hours")
    total_komplain: int = Field(..., alias="Total komplain")
    total_biaya_perbaikan: float = Field(..., alias="Total biaya perbaikan")
    
    class Config:
        # Allows accessing fields using their python names or their aliases
        populate_by_name = True
        
class RULPredictionResponse(BaseModel):
    asset_input: AssetInput
    predicted_rul: float
    
class ChatRequest(BaseModel):
    message: str
