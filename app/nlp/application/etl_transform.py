import re
from collections import Counter
from typing import List, Dict
from app.nlp.domain.models import ChatMessage

# --- Stopwords básicas (puedes ampliarlas luego si quieres) ---
DEFAULT_STOPWORDS = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
    "un", "para", "con", "no", "una", "su", "al", "lo", "como", "más", "pero",
    "sus", "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando",
    "muy", "sin", "sobre", "también", "me", "mi", "tengo", "te", "que"
}

class ETLTransformer:
    """
    Clase encargada de la **Transformación (Transform)** dentro del proceso ETL del chatbot.

    Rol:
    - Recibe los mensajes extraídos por el ETLExtractor.
    - Limpia el texto y genera estadísticas agregadas.
    """

    @staticmethod
    def clean_text(text: str) -> List[str]:
        """
        Limpia el texto y devuelve una lista de palabras significativas.
        """
        text = text.lower()
        text = re.sub(r"[^a-záéíóúñü\s]", "", text)
        words = text.split()
        return [w for w in words if w not in DEFAULT_STOPWORDS]

    def transform(self, messages: List[ChatMessage]) -> Dict:
        """
        Transforma los datos extraídos:
        - Limpia los textos
        - Calcula conteo de palabras y estadísticas básicas
        """
        print("⚙️ Transformando datos...")

        if not messages:
            print("⚠️ No se encontraron mensajes para transformar.")
            return {}

        # --- Unir todos los textos ---
        all_text = " ".join([msg.message for msg in messages if msg.message])
        words = self.clean_text(all_text)

        # --- Contar palabras más comunes ---
        word_counts = Counter(words)
        top_words = word_counts.most_common(10)

        # --- Métricas básicas ---
        total_messages = len(messages)
        avg_length = round(len(all_text.split()) / total_messages, 2)

        transformed_data = {
            "total_messages": total_messages,
            "avg_words_per_message": avg_length,
            "top_words": [{"word": w, "count": c} for w, c in top_words],
        }

        print(f"✅ {total_messages} mensajes transformados correctamente.")
        return transformed_data
