"""
chatbot_service.py

Chatbot estilo "psicólogo express mejorado".

Características:
- Lazy loading de módulos pesados (summary & sentimiento).
- Detección de emociones finas por lexicon (frustración, ansiedad, alegría, etc.).
- Uso del historial (memoria) para contextualizar respuestas.
- Respuestas empáticas y variadas adaptadas a la emoción detectada.
- Soporta resumen automático para textos largos (y resumen a petición).
- Guarda en BD tanto mensajes de usuario como respuestas del bot,
  almacenando además el campo `sentimiento` y `resumen` cuando aplica.

Notas:
- Esto es una solución práctica sin LLM. Si en el futuro quieres respuestas
  más ricas y específicas, la integración con un LLM (OpenAI/HuggingFace) sería el paso.
"""

import json
import random
import re
from typing import List, Tuple
from sqlalchemy.orm import Session
from app.nlp.domain.models import ChatMessage

# ====== Lexicon básico para emociones finas ======
# Palabras clave por emoción (muy simple, se puede ampliar)
EMOTION_KEYWORDS = {
    "frustración": ["frustrado", "frustrada", "frustración", "no sale", "no logro", "fallé", "fracaso"],
    "ansiedad": ["ansiedad", "ansioso", "ansiosa", "nervios", "angustia"],
    "estrés": ["estresado", "estresada", "estres", "estresante", "estresado/a", "estresarme"],
    "tristeza": ["triste", "tristeza", "lloré", "lloro", "deprimido", "deprimida"],
    "enfado": ["enojado", "enojada", "molesto", "molesta", "ira", "rabia"],
    "alegría": ["feliz", "alegre", "contento", "contenta", "emocionado", "alegría"],
    "cansancio": ["cansado", "cansada", "agotado", "agotada"],
    "soledad": ["solo", "sola", "solitario", "solitaria"],
    "injusticia": ["injusto", "injusta", "no valoran", "no me valoran"],
    "motivación": ["motivado", "motivada", "motivación", "con ganas"],
}

# Plantillas empáticas por emoción (variadas)
RESPONSE_TEMPLATES = {
    "frustración": [
        "Siento que estás frustrada/o. ¿Quieres contarme qué fue lo que más te molestó? 💬",
        "Parece que la frustración te está pesando. ¿Qué parte te gustaría resolver primero?"
    ],
    "ansiedad": [
        "Se nota cierta ansiedad en lo que dices. Respiremos un momento: ¿qué está ocupando tu mente ahora?",
        "La ansiedad puede ser muy incómoda. ¿Quieres contarme cuándo empezó o qué la desencadenó?"
    ],
    "estrés": [
        "El estrés puede agotarnos. ¿Hay algo que podamos priorizar o soltar por ahora?",
        "Siento que estás muy estresada/o. ¿Quieres que hagamos una pequeña lista de alivio?"
    ],
    "tristeza": [
        "Siento que te sientes triste. Gracias por confiar en contarme esto. ¿Qué parte te pesa más?",
        "La tristeza puede sentirse muy densa. ¿Quieres hablar sobre lo que la provoca?"
    ],
    "enfado": [
        "Parece que estás enojada/o. Es válido sentirlo. ¿Quieres que lo exploremos juntos para ver qué se puede hacer?",
        "La rabia puede indicarnos límites que fueron traspasados. ¿Qué pasó exactamente?"
    ],
    "alegría": [
        "¡Qué bueno! Me alegra escuchar eso 😊. ¿Qué fue lo que más disfrutaste?",
        "Se escucha alegría — ¡cuéntame más para celebrarlo contigo!"
    ],
    "cansancio": [
        "Noté cansancio en tu mensaje. ¿Has tenido tiempo para descansar o desconectar?",
        "El cansancio puede ser acumulativo. ¿Qué crees que podrías hacer hoy para recargar un poco?"
    ],
    "soledad": [
        "Siento que te sientes sola/o. Gracias por compartirlo. ¿Qué te haría sentir un poco más acompañada/o?",
        "La soledad pesa mucho. ¿Hay alguna persona con quien te gustaría hablar sobre esto?"
    ],
    "injusticia": [
        "Eso suena injusto y comprensible que te afecte. ¿Quieres contarme un ejemplo concreto?",
        "La sensación de no ser valorada puede doler bastante. ¿Qué te gustaría cambiar en esa situación?"
    ],
    "motivación": [
        "Se nota motivación — ¡eso es genial! ¿Qué paso siguiente te gustaría dar?",
        "Tener ganas es una señal poderosa. ¿Cómo podría ayudarte a canalizar esa energía?"
    ],
    # fallback neutral
    "neutral": [
        "Entiendo. ¿Quieres contarme más sobre eso?",
        "Ajá, te sigo… ¿qué más pasó?"
    ]
}


# ====== Helpers ======
def normalize_text(text: str) -> str:
    """Lowercase + remove extra spaces to ease matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


def detect_emotions_from_lexicon(text: str, top_k: int = 2) -> List[Tuple[str, int]]:
    """
    Devuelve una lista ordenada (emoción, score) detectadas a partir del lexicon.
    Score = cantidad de apariciones de palabras clave.
    """
    text_norm = normalize_text(text)
    scores = {}
    for emo, keys in EMOTION_KEYWORDS.items():
        s = 0
        for k in keys:
            if k in text_norm:
                s += text_norm.count(k)
        if s > 0:
            scores[emo] = s
    # ordenar por score descendente
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_k]


def aggregate_emotions(emotion_scores: List[Tuple[str, int]]) -> str:
    """Devuelve string compacto de emociones, o 'Neutral' si no hay."""
    if not emotion_scores:
        return "Neutral"
    return ", ".join([emo.capitalize() for emo, _ in emotion_scores])


def get_recent_user_messages(history: List[ChatMessage], n: int = 3) -> List[str]:
    """Extrae las últimas n mensajes enviados por el usuario en la sesión."""
    user_msgs = [m.message for m in history if m.sender == "user"]
    return user_msgs[-n:]


# ====== Persistencia ======
def save_message(session_id: str, sender: str, message: str, db: Session, resumen: str = None, sentimiento: str = None):
    """
    Guarda un mensaje (usuario o bot) en la base de datos y lo retorna.
    Mantiene compatibilidad con el modelo ChatMessage (resumen como string JSON).
    """
    chat = ChatMessage(
        session_id=session_id,
        sender=sender,
        message=message,
        resumen=resumen,
        sentimiento=sentimiento
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    return chat


# ====== Generador de respuesta empática ======
def generate_empathic_reply(detected_emotions: List[Tuple[str, int]], user_text: str, history: List[ChatMessage]) -> str:
    """
    Construye una respuesta empática basada en las emociones detectadas y el historial.
    - Si hay emociones detectadas, elige una plantilla acorde.
    - Si no, usa respuestas neutras y pide más contexto.
    """
    if detected_emotions:
        primary = detected_emotions[0][0]  # emoción principal (clave)
        templates = RESPONSE_TEMPLATES.get(primary, RESPONSE_TEMPLATES["neutral"])
        reply = random.choice(templates)

        # Añadir una línea de seguimiento que use parte del texto reciente para darle continuidad
        recent = get_recent_user_messages(history, n=2)
        if recent:
            snippet = recent[-1]
            # recortamos snippet para no exceder
            snippet_short = (snippet[:120] + "...") if len(snippet) > 120 else snippet
            follow = f" (por ejemplo: «{snippet_short}»)."
            # Insert follow if it doesn't make reply awkward
            reply = reply.rstrip("?") + " —" + follow
        return reply
    else:
        return random.choice(RESPONSE_TEMPLATES["neutral"])


# ====== Lógica principal ======
def process_message(session_id: str, user_text: str, db: Session):
    """
    Procesa el mensaje entrante del usuario:
    - Guarda el mensaje (user).
    - Carga historial y aplica detección de emociones finas.
    - Si el usuario pide 'sentimiento' devuelve el último sentimiento guardado.
    - Si el usuario pide 'resumir' o manda texto largo => genera resumen.
    - Para inputs normales => genera respuesta empática basada en emociones detectadas.
    - Guarda la respuesta del bot (con sentimiento o resumen cuando aplique).
    - Devuelve el objeto ChatMessage del bot (para que la API lo transforme a schema).
    """
    # Lazy load de servicios pesados (si se usan)
    from app.nlp.application.summary_service import resumir_texto  # resumir_texto puede devolver (str, float) en tu implementación

    # 1) Persistir mensaje del usuario
    save_message(session_id, "user", user_text, db)

    # 2) Recuperar historial (memoria)
    history = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.timestamp).all()

    lower = user_text.lower()

    # 3) Si el usuario pregunta por "sentimiento" (consulta de historial)
    if "sentimiento" in lower:
        last_sentiment = next(
            (msg.sentimiento for msg in reversed(history) if msg.sentimiento),
            None
        )
        if last_sentiment:
            bot_response = f"💡 El último sentimiento que detecté fue **{last_sentiment}**."
        else:
            bot_response = "❌ No encontré un sentimiento previo."

        chat = save_message(session_id, "bot", bot_response, db)
        return chat

    # 4) Si el usuario pide resumir textualmente
    if "resumir" in lower or "resume" in lower:
        bot_response = "Claro, pásame el texto a resumir."
        chat = save_message(session_id, "bot", bot_response, db)
        return chat

    # 5) Si el texto es largo -> resumir automáticamente
    if len(user_text.split()) > 30:
        # asumimos resumir_texto devuelve (resumen_texto, reduccion_percent)
        resumen_texto, reduccion = resumir_texto(user_text)

        resumen_dict = {
            "texto_original": user_text,
            "palabras_original": len(user_text.split()),
            "palabras_resumen": len(resumen_texto.split()),
            "reduccion": f"{reduccion}%"
        }

        bot_response = (
            f"📋 Resumen de lo que me contaste:\n{resumen_texto}\n\n"
            f"✨ Logré reducir el texto en {resumen_dict['reduccion']}."
        )

        chat = save_message(
            session_id,
            "bot",
            bot_response,
            db,
            resumen=json.dumps(resumen_dict)
        )
        return chat

    # 6) Detección de emociones finas por lexicon
    emotion_scores = detect_emotions_from_lexicon(user_text, top_k=3)
    detected_emotions = emotion_scores  # lista de (emo, score)
    detected_emotions_names = aggregate_emotions(emotion_scores)

    # 7) Generar respuesta empática basada en emociones detectadas y contexto
    # Si no detectamos emociones por lexicon, intentamos fallback simple: buscar sentimientos básicos
    if not detected_emotions:
        # importamos el analizador simple (si lo tienes) como respaldo para positivo/negativo/neutral
        try:
            from app.nlp.application.sentiment_service import analizar_sentimiento
            coarse = analizar_sentimiento(user_text)
            if coarse:
                if coarse.lower() == "positivo":
                    detected_emotions = [("alegría", 1)]
                elif coarse.lower() == "negativo":
                    detected_emotions = [("tristeza", 1)]
                else:
                    detected_emotions = []
        except Exception:
            # si no existe o falla, seguimos vacíos (neutral)
            detected_emotions = []

    bot_response = generate_empathic_reply(detected_emotions, user_text, history)

    # 8) Guardar respuesta y etiquetar sentimiento (si hay)
    sentimiento_field = aggregate_emotions(detected_emotions)
    chat = save_message(session_id, "bot", bot_response, db, sentimiento=sentimiento_field)

    return chat
