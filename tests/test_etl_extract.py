from sqlalchemy.orm import Session
from app.nlp.infrastructure.db import SessionLocal  # usa tu conexión de BD
from app.nlp.application.etl_extract import ETLExtractor

# ✅ 1. Conectarse a la base de datos
db: Session = SessionLocal()

# ✅ 2. Crear una instancia del extractor
extractor = ETLExtractor(db)

# ✅ 3. Extraer mensajes de una sesión (ajusta el session_id que tengas en tu BD)
mensajes = extractor.extract_messages(session_id="abc123")

# ✅ 4. Mostrar resultados en consola
print("\n=== MENSAJES EXTRAÍDOS ===")
for m in mensajes:
    print(f"[{m.timestamp}] {m.sender.upper()}: {m.message}")

print("\nTotal de mensajes extraídos:", len(mensajes))

# ✅ 5. Cerrar la conexión
db.close()
