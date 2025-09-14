from torchvision import transforms  # Para transformaciones de imágenes (rotar, normalizar, etc.)
from torchvision.datasets import ImageFolder  # Para leer imágenes organizadas en carpetas
from torch.utils.data import DataLoader, Subset  # DataLoader = hace batches, Subset = selecciona subconjunto
from pathlib import Path  # Manejo elegante de rutas de archivos/carpetas
from app.vision.utils.preprocess import preprocess_image  # Tu función personalizada para procesar imágenes
import torch  # Framework principal de deep learning


# 🔹 Definimos una clase para personalizar la forma en que se leen las imágenes
class CustomDataset(ImageFolder):
    def __init__(self, root, transform=None):
        # Llamamos al constructor de la clase padre (ImageFolder)
        # root = carpeta raíz donde están las imágenes
        super().__init__(root, transform=transform)

    # Esta función define cómo obtener UN solo elemento (imagen + etiqueta) del dataset
    def __getitem__(self, index):
        # Obtenemos la ruta de la imagen y la etiqueta (0 o 1 en este caso)
        path, label = self.samples[index]
        
        # Procesamos la imagen con tu función preprocess_image (convierte a tensor)
        tensor = preprocess_image(path, for_batch=False)
        
        # Devolvemos la imagen ya transformada y su etiqueta
        return tensor, label



# 🔹 Función para crear los DataLoaders (entrenamiento, validación, test)
def get_loaders(batch_size=32, subset_debug=False):
    # Localizamos la carpeta base (subimos dos niveles desde este archivo)
    BASE_DIR = Path(__file__).resolve().parent.parent
    
    # Definimos la ruta donde están los datos
    data_dir = BASE_DIR / "data" / "chest_xray"

    # Creamos datasets para cada partición de los datos
    train_dataset = CustomDataset(root=data_dir / "train")  # Imágenes de entrenamiento
    val_dataset = CustomDataset(root=data_dir / "val")      # Imágenes de validación
    test_dataset = CustomDataset(root=data_dir / "test")    # Imágenes de prueba final

    # 🔹 Opción: usar solo un subconjunto pequeño del dataset (útil para pruebas rápidas)
    if subset_debug:
        # Tomamos solo 200 imágenes para entrenar
        train_dataset = Subset(train_dataset, list(range(200)))
        # Tomamos solo 50 imágenes para validar
        val_dataset = Subset(val_dataset, list(range(50)))

    # Creamos DataLoaders que cargan los datos en lotes (batches)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)   # shuffle=True → mezcla cada época
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)      # validación NO necesita mezcla
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)    # test tampoco

    # Retornamos los 3 cargadores listos para usar en entrenamiento
    return train_loader, val_loader, test_loader
