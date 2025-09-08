from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from vision.utils.preprocess import preprocess_image

class CustomDataset(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, label = self.samples[index]
        tensor = preprocess_image(path, for_batch=False)  # sin batch
        return tensor, label

def get_loaders(data_dir="vision/data/chest_xray", batch_size=32):
    train_dataset = CustomDataset(root=f"{data_dir}/train")
    val_dataset = CustomDataset(root=f"{data_dir}/val")
    test_dataset = CustomDataset(root=f"{data_dir}/test")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
