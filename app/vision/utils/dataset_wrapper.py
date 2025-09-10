from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from app.vision.utils.preprocess import preprocess_image
import torch

class CustomDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, label = self.samples[index]
        tensor = preprocess_image(path, for_batch=False)  # ya convierte a tensor
        return tensor, label


def get_loaders(batch_size=32, subset_debug=False):
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data" / "chest_xray"

    train_dataset = CustomDataset(root=data_dir / "train")
    val_dataset = CustomDataset(root=data_dir / "val")
    test_dataset = CustomDataset(root=data_dir / "test")

    # 🔹 Subset para pruebas rápidas
    if subset_debug:
        train_dataset = Subset(train_dataset, list(range(200)))
        val_dataset = Subset(val_dataset, list(range(50)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
