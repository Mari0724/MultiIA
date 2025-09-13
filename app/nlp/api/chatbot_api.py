from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.nlp.infrastructure.db import get_db
from app.nlp.application.chatbot_service import process_message
from app.nlp.domain.schemas import ChatMessageCreate, ChatMessageResponse
from app.nlp.domain.models import ChatMessage

router = APIRouter(prefix="/chatbot", tags=["NLP"])

@router.post("/", response_model=ChatMessageResponse)
async def chatbot(message: ChatMessageCreate, db: Session = Depends(get_db)):
    response = process_message(message.session_id, message.text, db)
    return response

@router.get("/history/")
def get_chat_history(db: Session = Depends(get_db)):
    history = db.query(ChatMessage).order_by(ChatMessage.timestamp).all()
    return [
        {"id": msg.id, "sender": msg.sender, "text": msg.message, "timestamp": msg.timestamp}
        for msg in history
    ]
