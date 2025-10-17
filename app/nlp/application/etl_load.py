import json
import os

def load_data(data, output_path="data/processed/chatbot_data.json"):
    """
    Carga los datos transformados en un archivo JSON que servirá como base para el chatbot.
    """

    # Crear la carpeta 'data/processed' si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Guardar los datos en formato JSON con codificación UTF-8
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"✅ Datos cargados exitosamente en: {output_path}")
