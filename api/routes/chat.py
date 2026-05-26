from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
from core.database import get_db
from models.schemas import ChatRequest
from services.nlp_service import generate_chat_response

router = APIRouter()

@router.post("/chat")
def chat_endpoint(chat_request: ChatRequest, db: Session = Depends(get_db)):
    """
    Streaming response for NLP Chatbot / RAG.
    """
    # Return a streaming response from the NLP service
    return StreamingResponse(
        generate_chat_response(chat_request.message),
        media_type="text/event-stream"
    )
