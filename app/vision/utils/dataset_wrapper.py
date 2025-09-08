from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from app.vision.utils.preprocess import preprocess_image
from pathlib import Path

class CustomDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, label = self.samples[index]
        tensor = preprocess_image(path, for_batch=False)  # sin batch
        return tensor, label


def get_loaders(batch_size=32):
    # 📌 Construir ruta absoluta al dataset
    BASE_DIR = Path(__file__).resolve().parent.parent  # app/vision
    data_dir = BASE_DIR / "data" / "chest_xray"

    train_dataset = CustomDataset(root=data_dir / "train")
    val_dataset = CustomDataset(root=data_dir / "val")
    test_dataset = CustomDataset(root=data_dir / "test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
