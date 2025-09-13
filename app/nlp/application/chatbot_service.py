import json
import random
from sqlalchemy.orm import Session
from app.nlp.domain.models import ChatMessage
from app.nlp.application.sentiment_service import analizar_sentimiento
from app.nlp.application.summary_service import resumir_texto

def save_message(session_id: str, sender: str, message: str, db: Session, resumen: str = None, sentimiento: str = None):
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

def process_message(session_id: str, user_text: str, db: Session):
    # Guardar mensaje del usuario
    save_message(session_id, "user", user_text, db)

    # 👀 leer historial de la sesión
    history = db.query(ChatMessage).filter(
        ChatMessage.session_id == session_id
    ).order_by(ChatMessage.timestamp).all()

    # --- lógica con memoria ---
    if "sentimiento" in user_text.lower():
        # Buscar el último sentimiento registrado en la DB
        last_sentiment = next(
            (msg.sentimiento for msg in reversed(history) if msg.sentimiento),
            None
        )
        if last_sentiment:
            bot_response = f"💡 El último sentimiento que detecté fue **{last_sentiment}**."
        else:
            bot_response = "❌ No encontré un sentimiento previo."

    elif "resumir" in user_text.lower() or "resume" in user_text.lower():
        bot_response = "Claro, pásame el texto a resumir."

    elif len(user_text.split()) > 30:
        # ✅ ahora desempaquetamos bien la respuesta
        resumen_texto, reduccion = resumir_texto(user_text)

        resumen_dict = {
            "texto_original": user_text,
            "palabras_original": len(user_text.split()),
            "palabras_resumen": len(resumen_texto.split()),
            "reduccion": f"{100 - (len(resumen_texto.split())/len(user_text.split())*100):.2f}%"
        }

        bot_response = f"📋 Resumen de lo que me contaste:\n{resumen_texto}\n\n✨ Logré reducir el texto en {resumen_dict['reduccion']}."

        chat = save_message(
            session_id,
            "bot",
            bot_response,
            db,
            resumen=json.dumps(resumen_dict)  # ✅ guardado como JSON
        )
        return chat

    else:
        sentimiento = analizar_sentimiento(user_text)

        if sentimiento == "positivo":
            respuestas = [
                "😊 Me alegra escuchar eso. Cuéntame más de tu día.",
                "¡Eso suena genial! ¿Qué fue lo mejor que pasó?",
                "Se nota que estás feliz, y eso me alegra mucho."
            ]
        elif sentimiento == "negativo":
            respuestas = [
                "😔 Siento que no fue un buen momento. ¿Quieres hablar de ello?",
                "Entiendo, a veces los días son difíciles. Estoy aquí para escucharte.",
                "Debe haber sido duro… ¿qué crees que te haría sentir mejor?"
            ]
        else:  # neutral
            respuestas = [
                "🤔 Interesante, cuéntame más.",
                "Ajá, te sigo… ¿qué más pasó?",
                "Lo entiendo, y dime, ¿cómo te sientes con eso?"
            ]

        bot_response = random.choice(respuestas)

        chat = save_message(
            session_id,
            "bot",
            bot_response,
            db,
            sentimiento=sentimiento
        )
        return chat

    # Guardar y devolver (si no fue resumen o sentimiento directo)
    chat = save_message(session_id, "bot", bot_response, db)
    return chat
