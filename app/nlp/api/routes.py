from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.nlp.application.comentario_service import (
    crear_comentario, listar_comentarios, obtener_comentario, eliminar_comentario
)
from app.nlp.domain.schemas import ComentarioCreate, ComentarioResponse, TextoResumen
from app.nlp.application.summary_service import resumir_texto
from app.nlp.infrastructure.db import get_db
from app.nlp.api import chatbot_api   # 👈 lo importas para incluirlo

router = APIRouter(prefix="/nlp", tags=["NLP"])

# --- Rutas de comentarios ---
@router.post("/", response_model=ComentarioResponse)
def api_crear_comentario(comentario: ComentarioCreate, db: Session = Depends(get_db)):
    return crear_comentario(db, comentario.texto, comentario.sentimiento, comentario.resumen)

@router.get("/", response_model=list[ComentarioResponse])
def api_listar_comentarios(db: Session = Depends(get_db)):
    return listar_comentarios(db)

@router.get("/{comentario_id}", response_model=ComentarioResponse)
def api_obtener_comentario(comentario_id: int, db: Session = Depends(get_db)):
    comentario = obtener_comentario(db, comentario_id)
    if not comentario:
        raise HTTPException(status_code=404, detail="Comentario no encontrado")
    return comentario

@router.delete("/{comentario_id}")
def api_eliminar_comentario(comentario_id: int, db: Session = Depends(get_db)):
    comentario = eliminar_comentario(db, comentario_id)
    if not comentario:
        raise HTTPException(status_code=404, detail="Comentario no encontrado")
    return {"message": "Comentario eliminado", "id": comentario_id}

# --- Rutas de resumen ---
@router.post("/resumen")
def generar_resumen(payload: TextoResumen):
    """
    Genera un resumen a partir de un texto largo en español.
    """
    if not payload.texto:
        raise HTTPException(status_code=400, detail="Debes enviar un texto para resumir")
    return resumir_texto(payload.texto)

# --- Incluir chatbot ---
router.include_router(chatbot_api.router)
