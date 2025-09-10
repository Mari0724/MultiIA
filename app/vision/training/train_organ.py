# app/vision/training/train_organ.py
import torch
import torch.nn as nn
import torch.optim as optim
from app.vision.domain.organ_model import OrganClassifier
from app.vision.utils.dataset_wrapper import CustomDataset
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import os

def train_organ_model(epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 📂 Directorios base
    BASE_DIR = Path(__file__).resolve().parent.parent  # app/vision
    data_dir = BASE_DIR / "data" / "chest_xray"
    model_dir = BASE_DIR / "infrastructure" / "model"
    plots_dir = BASE_DIR / "infrastructure" / "plots"

    model_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_path = model_dir / "organ_cnn.pth"
    plot_path = plots_dir / "organ_training.png"

    # 📂 Dataset
    train_dataset = CustomDataset(root=data_dir / "train")
    val_dataset = CustomDataset(root=data_dir / "val")
    print(f"🔢 Imágenes en entrenamiento: {len(train_dataset)}")
    print(f"🔢 Imágenes en validación: {len(val_dataset)}")


    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model = OrganClassifier().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Validación
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.float().to(device).unsqueeze(1)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        val_losses.append(val_loss / len(val_loader))

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_losses[-1]:.4f}")

    # Guardar modelo
    torch.save(model.state_dict(), save_path)
    print(f"✅ Modelo de órgano guardado en {save_path}")

    # Graficar pérdidas
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Entrenamiento Clasificador de Órgano')
    plt.savefig(plot_path)
    plt.close()
    print(f"📊 Gráfica guardada en {plot_path}")

if __name__ == "__main__":
    train_organ_model()
