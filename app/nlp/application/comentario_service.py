from sqlalchemy.orm import Session
from app.nlp.domain.models import Comentario

def crear_comentario(db: Session, texto: str, sentimiento: str = None, resumen: str = None):
    """
    Crea un nuevo comentario en la base de datos.
    """
    nuevo = Comentario(texto=texto, sentimiento=sentimiento, resumen=resumen)
    db.add(nuevo)
    db.commit()
    db.refresh(nuevo)
    return nuevo

def listar_comentarios(db: Session):
    """
    Lista todos los comentarios.
    """
    return db.query(Comentario).all()

def obtener_comentario(db: Session, comentario_id: int):
    """
    Obtiene un comentario por su ID.
    """
    return db.query(Comentario).filter(Comentario.id == comentario_id).first()

def eliminar_comentario(db: Session, comentario_id: int):
    """
    Elimina un comentario por ID.
    """
    comentario = db.query(Comentario).filter(Comentario.id == comentario_id).first()
    if comentario:
        db.delete(comentario)
        db.commit()
    return comentario
