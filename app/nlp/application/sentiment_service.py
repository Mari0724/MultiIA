from pysentimiento import create_analyzer

# 📌 Cargamos el analizador de sentimientos ya preentrenado
analyzer = create_analyzer(task="sentiment", lang="es")

# 📌 Mapeamos a nuestras categorías personalizadas
MAPEO = {
    "POS": "Feliz",
    "NEG": "Negativo",
    "NEU": "Neutral"
}

def analizar_sentimiento(texto: str) -> str:
    """
    Analiza el sentimiento de un texto y lo traduce a nuestras categorías.
    """
    resultado = analyzer.predict(texto)
    return MAPEO.get(resultado.output, "Neutral")
