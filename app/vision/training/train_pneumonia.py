import torch
import torch.nn as nn
import torch.optim as optim
from app.vision.domain.pneumonia_model import SimpleCNN
from app.vision.utils.dataset_wrapper import get_loaders
from pathlib import Path
import matplotlib.pyplot as plt

def train_pneumonia_model(epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  #si tienes GPU con CUDA, usará la GPU. Si no, se queda en GPU 


    # 📂 Directorios base
    BASE_DIR = Path(__file__).resolve().parent.parent  # app/vision
    model_dir = BASE_DIR / "infrastructure" / "model"
    plots_dir = BASE_DIR / "infrastructure" / "plots"

    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_path = model_dir / "pneumonia_cnn.pth"
    plot_path = plots_dir / "pneumonia_training.png"

    # 1. Data
    """Llama a tu helper get_loaders.
            Devuelve tres objetos DataLoader:
            train_loader → para entrenar.
            val_loader → para validar.
            test_loader → para probar al final ("""
    train_loader, val_loader, test_loader = get_loaders()

    # 2. Modelo
    model = SimpleCNN().to(device)

    # 3. Loss y optimizador
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses, val_accuracies = [], [], []

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

        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)

        # Validación
        model.eval()
        val_loss = 0.0
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).int()

                correct += (preds == labels.int()).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader)
        accuracy = correct / total
        val_losses.append(val_loss)
        val_accuracies.append(accuracy)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {accuracy:.2f}")

    # Guardar modelo
    torch.save(model.state_dict(), save_path)
    print(f"✅ Modelo guardado en {save_path}")

    # Guardar gráfica
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training Metrics - Pneumonia CNN")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Gráfica guardada en {plot_path}")

if __name__ == "__main__":
    train_pneumonia_model()
