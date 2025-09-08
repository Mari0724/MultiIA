import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from app.vision.domain.pneumonia_model import SimpleCNN
from app.vision.utils.dataset_wrapper import get_loaders

MODEL_DIR = "vision/infrastructure/models"
PLOTS_DIR = "vision/infrastructure/plots"

def train_pneumonia_model(epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Crear carpetas si no existen
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    save_path = os.path.join(MODEL_DIR, "pneumonia_cnn.pth")

    # 1. Cargar datos
    train_loader, val_loader, test_loader = get_loaders()

    # 2. Modelo
    model = SimpleCNN().to(device)

    # 3. Loss y optimizador
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Listas para métricas
    train_losses = []
    val_losses = []
    val_accuracies = []

    # 4. Entrenamiento
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validación
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                preds = (outputs > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        accuracy = correct / total
        val_losses.append(avg_val_loss)
        val_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {accuracy:.4f}")

    # 5. Guardar modelo
    torch.save(model.state_dict(), save_path)
    print(f"✅ Modelo guardado en {save_path}")

    # 6. Graficar métricas
    plot_path = os.path.join(PLOTS_DIR, "pneumonia_training.png")
    plt.figure(figsize=(10, 5))

    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.legend()
    plt.title("Pérdida")

    # Precisión
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label="Val Accuracy", color="green")
    plt.legend()
    plt.title("Precisión Validación")

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.show()
    print(f"📊 Gráfica guardada en {plot_path}")

if __name__ == "__main__":
    train_pneumonia_model(epochs=5)
