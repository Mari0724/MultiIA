from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# 📌 Cargamos modelo preentrenado para resumen en español
MODEL_NAME = "mrm8488/bert2bert_shared-spanish-finetuned-summarization"

print("🔍 Cargando modelo de resumen de textos...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def resumir_texto(texto: str) -> dict:
    """
    Genera un resumen del texto y calcula el porcentaje de reducción.
    """
    # Resumen con HuggingFace
    resumen = summarizer(texto, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]

    # Contar palabras
    palabras_original = len(texto.split())
    palabras_resumen = len(resumen.split())

    # Evitamos división por cero
    reduccion = 0
    if palabras_original > 0:
        reduccion = round(((palabras_original - palabras_resumen) / palabras_original) * 100, 2)

    return {
        "texto_original": texto,
        "resumen": resumen,
        "palabras_original": palabras_original,
        "palabras_resumen": palabras_resumen,
        "reduccion": f"{reduccion}%"
    }
