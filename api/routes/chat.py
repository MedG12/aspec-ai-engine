from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
from core.database import get_db
from models.schemas import ChatRequest
from services.chat_service import stream_chat_response

router = APIRouter()

@router.post("/chat")
def chat_endpoint(chat_request: ChatRequest, db: Session = Depends(get_db)):
    """
    Streaming response for NLP Chatbot / RAG with sliding window and tools.
    """
    return StreamingResponse(
        stream_chat_response(db, chat_request),
        media_type="text/event-stream"
    )
