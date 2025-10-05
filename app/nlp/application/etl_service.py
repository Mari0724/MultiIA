import json
import re
from collections import Counter
from datetime import datetime
from typing import List, Optional, Tuple
from sqlalchemy.orm import Session
from app.nlp.domain.models import ChatMessage, ChatAnalytics

"""
    Este archivo contiene la lógica del **servicio ETL (Extract, Transform, Load)** para el chatbot.

    Rol dentro de la arquitectura:
    - Se encarga de extraer mensajes desde la base de datos, transformarlos en métricas útiles
    (conteo, sentimientos, palabras frecuentes, promedios), y cargar esos resultados en la tabla
    de `ChatAnalytics`.
    - Permite que el chatbot no solo responda, sino que también genere estadísticas de valor
    para análisis posteriores.

    Detalles:
    1. **Utilidades**:
    - `normalize_text`, `tokenize` y `top_n_words`: funciones para limpiar y analizar el texto.
    - Se eliminan stopwords comunes en español y se identifican las palabras más frecuentes.

    2. **Extract** (`extract_messages`):
    - Recupera los mensajes de la DB según un `session_id` o un rango de tiempo.
    - Los ordena cronológicamente.

    3. **Transform** (`transform_messages`):
    - Calcula métricas de la conversación:
        - Número de mensajes totales, de usuario y del bot.
        - Conteo de sentimientos (positivo, negativo, neutral).
        - Palabras más frecuentes usadas por el usuario.
        - Promedio de palabras por mensaje.
        - Intervalo de tiempo de la conversación (inicio y fin).
    - Devuelve un diccionario con estas métricas.

    4. **Load** (`load_metrics`):
    - Crea un registro en la tabla `ChatAnalytics` con los resultados de la transformación.
    - Lo guarda en la DB.

    5. **Orquestador** (`run_etl`):
    - Integra las tres fases: extraer, transformar y cargar.
    - Es el punto de entrada que se puede invocar desde la API o desde un cronjob para generar
        estadísticas de cualquier sesión de chat.

    Ventaja de esta capa:
    - Separa el análisis de datos (ETL) de la lógica del chatbot.
    - Permite reutilizar la información para reportes, dashboards o modelos de machine learning.
    - Si mañana cambiamos la forma de analizar sentimientos o palabras, solo modificamos
    la fase **Transform** sin alterar el resto.
"""

# pequeño set de stopwords en español (completa según necesidad)
DEFAULT_STOPWORDS = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para",
    "con","no","una","su","al","lo","como","más","pero","sus","le","ya","o",
    "este","sí","porque","esta","entre","cuando","muy","sin","sobre","también",
    "me","mi","tengo","te","que"
}

# ---------- UTILIDADES ----------
def normalize_text(text: str) -> str:
    text = text or ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\sáéíóúüñ]", " ", text)  # deja letras y acentos
    return text

def tokenize(text: str) -> List[str]:
    return [t for t in normalize_text(text).split() if t]

def top_n_words(tokens: List[str], stopwords=None, n=10):
    stop = stopwords or DEFAULT_STOPWORDS
    filtered = [t for t in tokens if t not in stop and len(t) > 1]
    counts = Counter(filtered)
    return counts.most_common(n)

# ---------- EXTRACT ----------
def extract_messages(db: Session, session_id: Optional[str] = None, since: Optional[datetime] = None):
    q = db.query(ChatMessage)
    if session_id:
        q = q.filter(ChatMessage.session_id == session_id)
    if since:
        q = q.filter(ChatMessage.timestamp >= since)
    q = q.order_by(ChatMessage.timestamp)
    return q.all()

# ---------- TRANSFORM ----------
def transform_messages(messages: List[ChatMessage]):
    # stats
    total = len(messages)
    user_msgs = [m for m in messages if m.sender == "user"]
    bot_msgs = [m for m in messages if m.sender == "bot"]

    # sentiment counts (robusto a mayúsculas/minúsculas)
    pos = neg = neu = 0
    for m in messages:
        s = (m.sentimiento or "").lower()
        if "pos" in s or "feliz" in s or "alegr" in s:
            pos += 1
        elif "neg" in s or "trist" in s or "enojo" in s or "frustr" in s or "estres" in s:
            neg += 1
        else:
            neu += 1

    # tokens y top words (solo mensajes de user)
    all_tokens = []
    for m in user_msgs:
        all_tokens.extend(tokenize(m.message))

    top = top_n_words(all_tokens, stopwords=DEFAULT_STOPWORDS, n=10)
    top_words_list = [{"word": w, "count": c} for w, c in top]

    # average words per message (user)
    avg_words = round((sum(len(tokenize(m.message)) for m in user_msgs) / (len(user_msgs) or 1)), 2)

    start_time = messages[0].timestamp if messages else None
    end_time = messages[-1].timestamp if messages else None

    metrics = {
        "total_messages": total,
        "user_messages": len(user_msgs),
        "bot_messages": len(bot_msgs),
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "avg_words": avg_words,
        "top_words": top_words_list,
        "start_time": start_time,
        "end_time": end_time,
    }
    return metrics

# ---------- LOAD ----------
def load_metrics(db: Session, session_id: Optional[str], metrics: dict) -> ChatAnalytics:
    analytics = ChatAnalytics(
        session_id=session_id,
        start_time=metrics["start_time"],
        end_time=metrics["end_time"],
        total_messages=metrics["total_messages"],
        user_messages=metrics["user_messages"],
        bot_messages=metrics["bot_messages"],
        positive=metrics["positive"],
        negative=metrics["negative"],
        neutral=metrics["neutral"],
        avg_words=metrics["avg_words"],
        top_words=metrics["top_words"]
    )
    db.add(analytics)
    db.commit()
    db.refresh(analytics)
    return analytics

# ---------- RUN ETL (orquestador) ----------
def run_etl(db: Session, session_id: Optional[str] = None, since: Optional[datetime] = None) -> ChatAnalytics:
    messages = extract_messages(db, session_id=session_id, since=since)
    metrics = transform_messages(messages)
    analytics = load_metrics(db, session_id, metrics)
    return analytics
