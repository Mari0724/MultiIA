import torch
from pathlib import Path  # Path = mapa para encontrar la casita del gato (nuestro modelo)


def save_model(model, path: str):
    """
    Guarda los pesos del modelo en el disco.
    - model: modelo de PyTorch que queremos guardar
    - path: ruta donde se guardará el modelo
    """
    
    # Nos aseguramos de que la carpeta donde guardaremos el modelo exista
    # Si no existe, la creamos (parents=True crea todas las carpetas necesarias)
    Path(path).parent.mkdir(parents=True, exist_ok=True)  # 🐱 preparamos la casita del gato
    
    # Guardamos solo los parámetros del modelo (pesos y sesgos)
    torch.save(model.state_dict(), path)  # guardamos al gato dentro de su casita


def load_model(model_cls, path: str):
    """
    Carga un modelo desde el disco.
    - model_cls: la clase del modelo que queremos cargar (ej. LinearRegressor)
    - path: ruta donde está guardado el modelo
    Retorna: instancia del modelo con los pesos cargados, lista para usar
    """
    
    p = Path(path)  # ubicamos la casita del gato
    if not p.exists():
        return None  # si no existe la casita, devolvemos None (no hay gato)
    
    # Creamos una instancia nueva del modelo
    model = model_cls()  # sacamos un gato nuevo del catálogo
    
    # Cargamos los pesos guardados en el archivo
    model.load_state_dict(torch.load(p))  # ponemos la experiencia guardada al gato
    
    # Ponemos el modelo en modo evaluación (ya no aprenderá, solo predice)
    model.eval()  #  el gato ya está listo para cazar sin entrenamiento
    
    return model  #  devolvemos al gato listo para usar
