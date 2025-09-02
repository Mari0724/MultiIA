# app/nlp/application/plot_sentiment.py
import os
import json
import matplotlib.pyplot as plt

LOGS_FILE = "./sentiment_model/trainer_state.json"
PLOTS_DIR = "app/prediction/infrastructure/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

if os.path.exists(LOGS_FILE):
    with open(LOGS_FILE, "r") as f:
        logs = json.load(f)

    logs_history = logs.get("log_history", [])

    steps = [i for i, log in enumerate(logs_history) if "loss" in log]
    losses = [log["loss"] for log in logs_history if "loss" in log]
    eval_losses = [log.get("eval_loss") for log in logs_history if "eval_loss" in log]
    eval_acc = [log.get("eval_accuracy") for log in logs_history if "eval_accuracy" in log]

    # Gráfico de pérdida
    plt.figure(figsize=(8,6))
    plt.plot(steps, losses, label="Train Loss")
    if eval_losses:
        plt.plot(steps[:len(eval_losses)], eval_losses, label="Eval Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Training & Eval Loss")
    plt.legend()
    plt.savefig(f"{PLOTS_DIR}/loss_curve.png")
    plt.close()

    # Gráfico de accuracy
    if eval_acc:
        plt.figure(figsize=(8,6))
        plt.plot(eval_acc, label="Eval Accuracy", color="green")
        plt.xlabel("Evaluation Step")
        plt.ylabel("Accuracy")
        plt.title("Evaluation Accuracy")
        plt.legend()
        plt.savefig(f"{PLOTS_DIR}/accuracy_curve.png")
        plt.close()

    print(f"📊 Gráficas guardadas en {PLOTS_DIR}")
else:
    print("⚠️ No se encontró trainer_state.json. Entrena primero el modelo.")
