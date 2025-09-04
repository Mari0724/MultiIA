import matplotlib.pyplot as plt
from sqlalchemy.orm import Session
from app.nlp.infrastructure.db import SessionLocal
from app.nlp.application.comentario_service import listar_comentarios
import os

# 📌 Carpeta para guardar las gráficas
PLOTS_DIR = "app/nlp/infrastructure/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def generar_grafica_sentimientos():
    db: Session = SessionLocal()
    comentarios = listar_comentarios(db)

    # Contar sentimientos
    conteo = {"Feliz": 0, "Negativo": 0, "Neutral": 0}
    for c in comentarios:
        if c.sentimiento in conteo:
            conteo[c.sentimiento] += 1

    # 📊 Gráfica de barras
    plt.figure(figsize=(6,4))
    plt.bar(conteo.keys(), conteo.values())
    plt.title("Distribución de Sentimientos en Comentarios")
    plt.ylabel("Cantidad de comentarios")
    plt.savefig(f"{PLOTS_DIR}/sentimientos_bar.png")
    plt.close()

    # 📊 Gráfica de pastel
    plt.figure(figsize=(6,6))
    plt.pie(conteo.values(), labels=conteo.keys(), autopct="%1.1f%%")
    plt.title("Distribución de Sentimientos en Comentarios")
    plt.savefig(f"{PLOTS_DIR}/sentimientos_pie.png")
    plt.close()

    print(f"✅ Gráficas guardadas en {PLOTS_DIR}/")

if __name__ == "__main__":
    generar_grafica_sentimientos()
