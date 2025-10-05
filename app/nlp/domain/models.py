from sqlalchemy import Column, Integer, String, Text, DateTime, Float, func
from app.nlp.infrastructure.db import Base
from sqlalchemy.dialects.postgresql import JSON

class Comentario(Base):
    __tablename__ = "comentarios"

    id = Column(Integer, primary_key=True, index=True)
    texto = Column(Text, nullable=False)   # comentario original
    sentimiento = Column(String(20))       # feliz, triste, enojado, neutral
    resumen = Column(Text)                 # resumen del comentario
    fecha = Column(DateTime(timezone=True), server_default=func.now())  # fecha automática

class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    sender = Column(String(20), nullable=False)   # "user" o "bot"
    message = Column(Text, nullable=False)        # texto del mensaje
    sentimiento = Column(String(20), nullable=True)  # 😀 feliz, 😡 enojado...
    resumen = Column(Text, nullable=True)            # texto resumido
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class ChatAnalytics(Base):
    __tablename__ = "chat_analytics"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=True)  # si null -> agregado global
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    total_messages = Column(Integer, nullable=False, default=0)
    user_messages = Column(Integer, nullable=False, default=0)
    bot_messages = Column(Integer, nullable=False, default=0)
    positive = Column(Integer, nullable=False, default=0)
    negative = Column(Integer, nullable=False, default=0)
    neutral = Column(Integer, nullable=False, default=0)
    avg_words = Column(Float, nullable=True)
    top_words = Column(JSON, nullable=True)   # lista de {word:count}
    created_at = Column(DateTime(timezone=True), server_default=func.now())