from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from app.nlp.infrastructure.db import get_db
from app.nlp.application.etl_service import run_etl
from app.nlp.domain.models import ChatAnalytics

# Creamos un router de FastAPI con prefijo /nlp/etl
# Esto agrupa todas las rutas relacionadas con la ETL
router = APIRouter(prefix="/nlp/etl", tags=["ETL"])


# ----------------- ENDPOINT PARA EJECUTAR LA ETL -----------------
@router.post("/run")
def api_run_etl(
    session_id: str | None = None,   # opcional: ID de la sesión de chat
    since: str | None = None,        # opcional: filtrar mensajes desde cierta fecha
    db: Session = Depends(get_db)    # conexión a la base de datos
):
    """
    Ejecuta ETL:
    - session_id: opcional, si se pasa, ETL solo para esa sesión.
    - since: opcional, ISO datetime para filtrar mensajes recientes.
    """

    # Convertimos el string "since" a datetime si viene en la request
    since_dt = None
    if since:
        try:
            since_dt = datetime.fromisoformat(since)
        except Exception:
            # Si el formato de fecha no es válido, devolvemos un error HTTP
            raise HTTPException(status_code=400, detail="since debe ser ISO datetime")

    # Ejecutamos la ETL (Extract → Transform → Load)
    analytics = run_etl(db, session_id=session_id, since=since_dt)

    # Retornamos un resumen de las métricas generadas
    return {
        "id": analytics.id,
        "session_id": analytics.session_id,
        "created_at": analytics.created_at,
        "total_messages": analytics.total_messages,
        "positive": analytics.positive,
        "negative": analytics.negative,
        "neutral": analytics.neutral,
        "avg_words": analytics.avg_words,
        "top_words": analytics.top_words
    }


# ----------------- ENDPOINT PARA CONSULTAR UN REPORTE -----------------
@router.get("/report/{analytics_id}")
def api_get_analytics(analytics_id: int, db: Session = Depends(get_db)):
    # Buscamos en la base de datos el análisis guardado con ese ID
    a = db.query(ChatAnalytics).filter(ChatAnalytics.id == analytics_id).first()

    # Si no existe, devolvemos un error 404
    if not a:
        raise HTTPException(404, "No encontrado")

    # Retornamos todos los detalles de ese análisis
    return {
        "id": a.id,
        "session_id": a.session_id,
        "start_time": a.start_time,
        "end_time": a.end_time,
        "total_messages": a.total_messages,
        "user_messages": a.user_messages,
        "bot_messages": a.bot_messages,
        "positive": a.positive,
        "negative": a.negative,
        "neutral": a.neutral,
        "avg_words": a.avg_words,
        "top_words": a.top_words,
        "created_at": a.created_at
    }
