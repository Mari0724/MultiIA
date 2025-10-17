from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from contextlib import contextmanager

# 🐱 Declarative_base = la camita donde se van a acostar las clases/tablas
Base = declarative_base()

# Configuración de la conexión
DATABASE_URL = "postgresql+psycopg2://postgres:123456@localhost:5432/nlp"

# Engine = motor que conecta Python ↔ PostgreSQL
engine = create_engine(DATABASE_URL, echo=True)

# Session = la cajita de arena donde los gatos juegan (cada query va dentro de una sesión)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

class DBConnection:
    """
    Clase para manejar la conexión a la base de datos.
    Abre y cierra la sesión automáticamente.
    """
    def __init__(self):
        self.session = SessionLocal()

    def __enter__(self):
        return self.session

    def __exit__(self, exc_type, exc_value, traceback):
        self.session.close()
