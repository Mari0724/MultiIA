from typing import Tuple

def resumir_texto(texto: str) -> Tuple[str, float]:
    """
    Resume un texto eliminando redundancias, manteniendo ortografía
    y generando frases compactas.

    Retorna:
        resumen (str): Texto resumido.
        reduccion (float): Porcentaje de reducción.
    """
    import re

    # 1. Limpiar texto (espacios extra, saltos de línea)
    texto = re.sub(r"\s+", " ", texto.strip())

    # 2. Dividir en oraciones
    oraciones = re.split(r"(?<=[.!?])\s+", texto)

    # 3. Seleccionar frases más relevantes (longitud media y que contengan verbos comunes)
    verbos_clave = ["fui", "tuve", "hice", "organicé", "me reuní", "levanté", "trabajé"]
    resumen_oraciones = [
        o for o in oraciones if any(v in o.lower() for v in verbos_clave)
    ]

    # Si no encuentra nada, tomar la primera y la última oración
    if not resumen_oraciones:
        resumen_oraciones = [oraciones[0], oraciones[-1]]

    # 4. Armar resumen compacto en una sola frase fluida
    resumen = " ".join(resumen_oraciones)
    resumen = resumen.replace("  ", " ")

    # 5. Calcular porcentaje de reducción
    reduccion = 100 * (1 - len(resumen) / len(texto))

    return resumen, round(reduccion, 2)
