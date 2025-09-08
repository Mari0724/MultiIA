import cv2

def draw_xray_annotation(img_path: str, is_chest: bool, prediction: str, confidence: float):
    """Dibuja texto y recuadro en la radiografía."""
    img = cv2.imread(img_path)

    if not is_chest:
        text = "NO ES TORAX ❌"
        color = (0, 0, 255)  # rojo
    else:
        text = f"{prediction} ({confidence:.2f})"
        color = (0, 255, 0) if prediction == "Normal" else (0, 0, 255)

    # Añadir texto en la parte superior
    cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1.2, color, 3, cv2.LINE_AA)

    return img
