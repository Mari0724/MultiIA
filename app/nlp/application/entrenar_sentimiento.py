from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1) Dataset base
dataset = load_dataset("go_emotions")

id2label = {0: "Feliz", 1: "Neutral", 2: "Enojado", 3: "Triste"}
label2id = {v: k for k, v in id2label.items()}

# 2) Simplificación de etiquetas
mapa = {
    "joy": "Feliz",
    "neutral": "Neutral",
    "anger": "Enojado",
    "sadness": "Triste"
}

# definimos al inicio, después de cargar dataset
label_names = dataset["train"].features["labels"].feature.names

def simplificar_etiqueta(example):
    etiquetas = example["labels"]

    if not etiquetas:  # Si está vacío
        return {"labels": label2id["Neutral"]}

    # Tomamos la primera etiqueta (GoEmotions es multilabel)
    emocion = label_names[etiquetas[0]]

    # Mapeamos a nuestras 4 clases
    clase = mapa.get(emocion, "Neutral")
    return {"labels": label2id[clase]}

dataset_simple = dataset.map(simplificar_etiqueta)

# 3) Tokenizer y modelo
MODEL_NAME = "bert-base-multilingual-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenizar(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=128)

dataset_token = dataset_simple.map(tokenizar, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    id2label=id2label,
    label2id=label2id
)

# 4) Funciones métricas
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# 5) Entrenamiento
training_args = TrainingArguments(
    output_dir="./sentiment_model",
    evaluation_strategy="epoch",   # 👈 evaluación al final de cada época
    save_strategy="epoch",         # 👈 guarda modelo cada época
    logging_dir="./logs",
    num_train_epochs=2,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    weight_decay=0.01
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset_token["train"].select(range(2000)),  # ⚡ dataset reducido
    eval_dataset=dataset_token["validation"].select(range(500)),
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()

# 👇 Guardar modelo completo con tokenizer y config
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")