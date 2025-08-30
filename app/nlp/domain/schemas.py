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
        orm_mode = True  # permite convertir de SQLAlchemy a Pydantic
