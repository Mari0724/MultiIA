import os
from app.nlp.infrastructure.db import SessionLocal

# 📌 Carpeta para guardar las gráficas
PLOTS_DIR = "app/nlp/infrastructure/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def generar_graficas(save_plot=True):
    """
    Genera gráficas de distribución de sentimientos en los comentarios.
    Si save_plot=False, no guarda imágenes (para los tests).
    """
    from app.nlp.domain.models import Comentario  # 👈 import solo cuando se llama
    import matplotlib.pyplot as plt              # 👈 import solo cuando se llama
    from collections import Counter              # 👈 igual aquí

    db = SessionLocal()
    comentarios = db.query(Comentario).all()
    db.close()

    sentimientos = [c.sentimiento for c in comentarios if c.sentimiento]

    if not sentimientos:
        return {"error": "No hay comentarios con sentimientos"}

    # Conteo de cada sentimiento
    conteo = Counter(sentimientos)

    # 📊 Gráfica de barras
    plt.figure(figsize=(6, 6))
    plt.bar(conteo.keys(), conteo.values(), color=["green", "red", "blue"])
    plt.title("Distribución de Sentimientos en Comentarios")
    plt.xlabel("Sentimiento")
    plt.ylabel("Cantidad")

    if save_plot:
        path = os.path.join(PLOTS_DIR, "sentimientos.png")
        plt.savefig(path)
        plt.close()
        return {"msg": f"📊 Gráfico guardado en {path}", "conteo": dict(conteo)}

    plt.close()
    return {"conteo": dict(conteo)}