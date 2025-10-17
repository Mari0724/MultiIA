from datetime import datetime
from app.nlp.application.etl_extract import ETLExtractor

# 🧠 Simulamos una "falsa base de datos" para probar la lógica
class FakeDB:
    def __init__(self):
        self.data = [
            {"session_id": "abc123", "text": "Hoy fue un día duro", "timestamp": datetime(2025, 10, 15, 10, 0)},
            {"session_id": "abc123", "text": "Estoy muy cansada", "timestamp": datetime(2025, 10, 15, 10, 5)},
            {"session_id": "abc123", "text": "Pero también aprendí cosas nuevas", "timestamp": datetime(2025, 10, 15, 10, 10)},
        ]

    def query(self, model):
        # devolvemos un objeto con métodos encadenables tipo SQLAlchemy
        return self

    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args, **kwargs):
        return self

    def all(self):
        return self.data


# 🧩 Creamos el extractor con nuestra base de datos simulada
fake_db = FakeDB()
extractor = ETLExtractor(db=fake_db)

# 🧪 Extraemos los mensajes
messages = extractor.extract_messages(session_id="abc123")

print("Mensajes extraídos:")
for m in messages:
    print(f"- {m['timestamp']}: {m['text']}")
