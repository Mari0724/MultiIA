from sqlalchemy.orm import Session
from app.nlp.domain.models import Comentario
from app.nlp.application.sentiment_service import analizar_sentimiento  # 👈 Importamos el modelo

def crear_comentario(db: Session, texto: str, sentimiento: str = None, resumen: str = None):
    """
    Crea un nuevo comentario en la base de datos.
    Si no se pasa sentimiento, lo calcula automáticamente usando el modelo.
    """
    # Si no viene sentimiento manual, lo calculamos
    if not sentimiento:
        sentimiento = analizar_sentimiento(texto)

    # 1) Creamos un objeto Comentario (aún NO está en la BD)
    nuevo = Comentario(texto=texto, sentimiento=sentimiento, resumen=resumen)

    # 2) Lo agregamos a la sesión
    db.add(nuevo)

    # 3) Guardamos en la BD
    db.commit()

    # 4) Refrescamos para obtener ID y fecha generados
    db.refresh(nuevo)

    # 5) Lo devolvemos
    return nuevo


def listar_comentarios(db: Session):
    """Lista todos los comentarios."""
    return db.query(Comentario).all()


def obtener_comentario(db: Session, comentario_id: int):
    """Obtiene un comentario por su ID."""
    return db.query(Comentario).filter(Comentario.id == comentario_id).first()


def eliminar_comentario(db: Session, comentario_id: int):
    """Elimina un comentario por ID."""
    comentario = db.query(Comentario).filter(Comentario.id == comentario_id).first()
    if comentario:
        db.delete(comentario)
        db.commit()
    return comentario
