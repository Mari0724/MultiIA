from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Modelo en español para sentimiento
MODEL_NAME = "pysentimiento/bert-base-spanish-uncased"

# Cargamos el tokenizer y modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

# Creamos un pipeline de análisis de sentimientos
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Probar
texto = "Me encantó la película, fue increíble."
resultado = sentiment_pipeline(texto)
print(resultado)
