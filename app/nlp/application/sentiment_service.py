from pysentimiento import create_analyzer

# Cache global (solo se inicializa la primera vez que se usa)
_analyzer = None  

MAPEO = {
    "POS": "Feliz",
    "NEG": "Negativo",
    "NEU": "Neutral"
}

def analizar_sentimiento(texto: str) -> str:
    """
    Analiza el sentimiento de un texto y lo traduce a nuestras categorías.
    Se inicializa el modelo solo cuando se necesite (Lazy Loading).
    """
    global _analyzer
    if _analyzer is None:
        _analyzer = create_analyzer(task="sentiment", lang="es")  # 👈 solo se carga la primera vez
    
    resultado = _analyzer.predict(texto)
    return MAPEO.get(resultado.output, "Neutral")
