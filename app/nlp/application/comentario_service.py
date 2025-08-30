from sqlalchemy.orm import Session
from app.nlp.domain.models import Comentario

def crear_comentario(db: Session, texto: str, sentimiento: str = None, resumen: str = None):
    """
    Crea un nuevo comentario en la base de datos.
    """
    # 1) Creamos un objeto Comentario (aún NO está en la BD)
    nuevo = Comentario(texto=texto, sentimiento=sentimiento, resumen=resumen)

    # 2) Lo agregamos a la sesión (cajita de arena)
    db.add(nuevo)

    # 3) Enviamos los cambios a la BD (lo guardamos de verdad)
    db.commit()

    # 4) Refrescamos para que tenga valores generados por la BD (id, fecha)
    db.refresh(nuevo)

    # 5) Lo devolvemos (ya con id y fecha)
    return nuevo


def listar_comentarios(db: Session):
    """
    Lista todos los comentarios.
    """
    # SELECT * FROM comentarios;
    return db.query(Comentario).all()


def obtener_comentario(db: Session, comentario_id: int):
    """
    Obtiene un comentario por su ID.
    """
    # SELECT * FROM comentarios WHERE id = :id LIMIT 1;
    return db.query(Comentario).filter(Comentario.id == comentario_id).first()


def eliminar_comentario(db: Session, comentario_id: int):
    """
    Elimina un comentario por ID (delete físico).
    """
    # 1) Buscamos el comentario
    comentario = db.query(Comentario).filter(Comentario.id == comentario_id).first()

    # 2) Si existe, lo borramos
    if comentario:
        db.delete(comentario)  # Marca el objeto para eliminar
        db.commit()            # Ejecuta DELETE en la BD

    # 3) Devolvemos el que se eliminó (o None si no existía)
    return comentario
