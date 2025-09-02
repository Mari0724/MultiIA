from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# 📌 Usamos el modelo entrenado
MODEL_PATH = "./sentiment_model"

print("🔍 Cargando modelo de sentimiento entrenado...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

sentiment_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

def analizar_sentimiento(texto: str) -> str:
    resultado = sentiment_pipeline(texto)[0]
    return resultado["label"]  # "Feliz", "Triste", "Enojado", "Neutral"
