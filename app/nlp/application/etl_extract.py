from datetime import datetime
from typing import Optional, List
from sqlalchemy.orm import Session
from app.nlp.domain.models import ChatMessage


class ETLExtractor:
    """
    Clase encargada de la **Extracción (Extract)** dentro del proceso ETL del chatbot.
    """

    def __init__(self, db: Session):
        """
        Constructor del extractor.
        - db (Session): sesión activa de SQLAlchemy.
        """
        self.db = db

    def extract(self, session_id: Optional[str] = None, since: Optional[datetime] = None):
        """
        Método público que ejecuta la extracción de mensajes desde la base de datos.
        """
        print("📥 Extrayendo datos desde la base de datos...")
        return self.extract_messages(session_id=session_id, since=since)

    def extract_messages(self, session_id: Optional[str] = None, since: Optional[datetime] = None) -> List[ChatMessage]:
        """
        Extrae mensajes del chatbot desde la base de datos.

        Parámetros:
        - session_id (str, opcional): ID de sesión para filtrar los mensajes.
        - since (datetime, opcional): fecha desde la cual se desean extraer los mensajes.

        Retorna:
        - Lista de objetos ChatMessage (ordenados cronológicamente).
        """
        query = self.db.query(ChatMessage)

        if session_id:
            query = query.filter(ChatMessage.session_id == session_id)
        if since:
            query = query.filter(ChatMessage.timestamp >= since)

        query = query.order_by(ChatMessage.timestamp)
        messages = query.all()

        print(f"✅ {len(messages)} mensajes extraídos.")
        return messages
