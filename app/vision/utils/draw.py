import cv2  # 📦 Librería OpenCV → permite leer imágenes, dibujar texto, rectángulos, etc.

# 🔹 Función que dibuja anotaciones sobre una radiografía
def draw_xray_annotation(img_path: str, is_chest: bool, prediction: str, confidence: float):
    """
    Dibuja texto sobre la radiografía dependiendo del resultado.
    
    Parámetros:
    - img_path (str): ruta de la imagen a procesar.
    - is_chest (bool): indica si la imagen realmente es de tórax.
    - prediction (str): predicción del modelo ("Normal" o "Neumonia").
    - confidence (float): nivel de confianza del modelo (0.0 a 1.0).
    
    Devuelve:
    - img: la imagen con el texto/anotación dibujada.
    """

    #  Leemos la imagen desde la ruta usando OpenCV
    img = cv2.imread(img_path)

    #  Caso 1: la imagen NO es de tórax
    if not is_chest:
        text = "NO ES TORAX ❌"  # mensaje directo
        color = (0, 0, 255)      # rojo → en formato BGR (Blue, Green, Red)

    #  Caso 2: la imagen SÍ es de tórax
    else:
        # Mostramos la predicción con su confianza (ej: "Normal (0.95)")
        text = f"{prediction} ({confidence:.2f})"

        # Verde si es "Normal", rojo si es "Neumonia"
        color = (0, 255, 0) if prediction == "Normal" else (0, 0, 255)

    # 🖍️ Dibujamos el texto en la parte superior de la imagen
    cv2.putText(
        img,                # imagen donde escribir
        text,               # texto a mostrar
        (20, 40),           # coordenadas (x=20, y=40) → esquina superior izquierda
        cv2.FONT_HERSHEY_SIMPLEX,  # tipo de fuente
        1.2,                # tamaño del texto
        color,              # color del texto (verde o rojo)
        3,                  # grosor de la línea
        cv2.LINE_AA         # suavizado de bordes (Anti-Aliasing)
    )

    #  Devolvemos la imagen ya modificada (NO se guarda aquí, solo se devuelve en memoria)
    return img
