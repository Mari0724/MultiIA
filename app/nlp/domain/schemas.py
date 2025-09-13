from pydantic import BaseModel
from datetime import datetime

# 📝 Para recibir comentarios (request)
class ComentarioCreate(BaseModel):
    texto: str
    sentimiento: str | None = None
    resumen: str | None = None

# 📤 Para devolver comentarios (response)
class ComentarioResponse(BaseModel):
    id: int
    texto: str
    sentimiento: str | None = None
    resumen: str | None = None
    fecha: datetime

class Config:
    from_attributes = True

# 📝 Para recibir mensajes de chat (request)
class ChatMessageCreate(BaseModel):
    session_id: str # 👈 necesario para continuar conversación
    text: str  # 👈 ahora sí coincide

# 📤 Para devolver mensajes de chat (response)
class ChatMessageResponse(BaseModel):
    id: int
    sender: str
    message: str
    timestamp: datetime

    class Config:
        from_attributes = True

# 📋 Para servicio de resumen
class TextoResumen(BaseModel):
    texto: str