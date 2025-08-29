from app.nlp.infrastructure.db import Base, engine
from app.nlp.domain.models import Comentario

# 🐱 Esto crea todas las tablas en PostgreSQL
print("Creando tablas en la BD...")
Base.metadata.create_all(bind=engine)
print("Tablas creadas ✅")
