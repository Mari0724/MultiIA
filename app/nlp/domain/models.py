from sqlalchemy import Column, Integer, String, Text, DateTime, func
from app.nlp.infrastructure.db import Base

class Comentario(Base):
    __tablename__ = "comentarios"

    id = Column(Integer, primary_key=True, index=True)
    texto = Column(Text, nullable=False)   # comentario original
    sentimiento = Column(String(20))       # feliz, triste, enojado, neutral
    resumen = Column(Text)                 # resumen del comentario
    fecha = Column(DateTime(timezone=True), server_default=func.now())  # fecha automática
