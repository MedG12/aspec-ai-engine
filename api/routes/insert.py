from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from core.database import get_db
from models.schemas import InsertResponse, BatchInsertRequest
from services.vector_db_service import batch_insert_and_embed, get_collection_count

router = APIRouter()


@router.post("/insert", response_model=InsertResponse)
def insert_endpoint(request: BatchInsertRequest, db: Session = Depends(get_db)):
    """
    Endpoint untuk insert batch data maintenance ke MySQL, melakukan indexing 
    ke ChromaDB, lalu set is_embedded = TRUE untuk data yang berhasil diindeks.
    """
    try:
        result = batch_insert_and_embed(db, request)
        return InsertResponse(
            success=True,
            mode=result["mode"],
            indexed=result["indexed"],
            total_in_db=result["total_in_db"],
            message=result["message"]
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Gagal melakukan insert dan indexing: {str(e)}"
        )


@router.get("/insert/status")
def insert_status_endpoint():
    """
    Cek status ChromaDB collection: berapa dokumen yang sudah terindeks.
    """
    try:
        count = get_collection_count()
        return {
            "collection_count": count,
            "is_indexed": count > 0,
            "message": f"ChromaDB memiliki {count} dokumen terindeks." if count > 0 else "ChromaDB belum diindeks."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Gagal mengambil status ChromaDB: {str(e)}"
        )
