# pip install pysentimiento

from pysentimiento import create_analyzer

# Crea el analizador de sentimientos en español
analyzer = create_analyzer(task="sentiment", lang="es")

def analizar_sentimiento(texto: str) -> str:
    resultado = analyzer.predict(texto)
    label = resultado.output  # "POS", "NEG" o "NEU"

    if label == "POS":
        return "Feliz"
    elif label == "NEG":
        return "Negativo"
    else:
        return "Neutral"
