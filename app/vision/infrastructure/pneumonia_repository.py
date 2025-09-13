# Importa la clase Path del módulo pathlib para manejar rutas de archivos de manera orientada a objetos.
# Es más robusto y fácil de usar que las cadenas de texto.
from pathlib import Path 

# Importa el módulo cv2, que es la biblioteca de código abierto de visión por computadora (OpenCV).
# Se usa para guardar la imagen procesada en el disco.
import cv2

class PneumoniaRepository:
    def __init__(self, base_dir: Path = None):
        """
        Constructor de la clase. Se ejecuta al crear un nuevo objeto PneumoniaRepository.

        Args:
            base_dir (Path, opcional): La ruta base para almacenar las imágenes.
            Si no se proporciona (es None), se calcula automáticamente.
        """
        # Si no se especifica una ruta base, se calcula una ruta predeterminada.
        if base_dir is None:
            # Obtiene la ruta del archivo actual (__file__), la resuelve a una ruta absoluta,
            # y sube tres niveles de directorio para llegar a la raíz del proyecto.
            # Esto hace que la ruta funcione sin importar desde dónde se ejecute el script.
            root_dir = Path(__file__).resolve().parent.parent.parent 
            # Construye la ruta completa a la carpeta de cargas (uploads) dentro del proyecto.
            base_dir = root_dir / "app" / "vision" / "uploads"

        # Asigna la ruta base como un atributo de la instancia (self.base_dir).
        self.base_dir = base_dir
        # Define la ruta para las imágenes "raw" (sin procesar).
        self.raw_dir = base_dir / "raw"
        # Define la ruta para las imágenes procesadas de rayos X.
        self.proc_dir = base_dir / "xray_proc"

        # Crea las carpetas "raw" y "xray_proc" si no existen.
        # parents=True: Crea las carpetas padres si no existen.
        # exist_ok=True: Evita un error si la carpeta ya existe.
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.proc_dir.mkdir(parents=True, exist_ok=True)

    async def save_raw(self, file, filename: str) -> Path:
        """
        Guarda un archivo subido de manera asíncrona en la carpeta 'raw'.

        Args:
            file: El objeto de archivo subido (típicamente de un framework como FastAPI).
            filename (str): El nombre del archivo con su extensión (ej. "imagen.jpg").

        Returns:
            str: La ruta completa del archivo guardado como una cadena de texto.
        """
        # Combina la ruta de la carpeta 'raw' con el nombre del archivo.
        file_path = self.raw_dir / filename

        # Lee el contenido del archivo subido de forma asíncrona.
        # El 'await' pausa la ejecución hasta que la lectura se complete,
        # lo que permite al programa realizar otras tareas mientras espera.
        content = await file.read()

        # Abre el archivo en modo de escritura binaria ('wb').
        # El 'with' asegura que el archivo se cierre automáticamente al salir del bloque.
        with open(file_path, "wb") as buffer:
            # Escribe el contenido del archivo en el búfer.
            buffer.write(content)

        # Devuelve la ruta del archivo guardado como una cadena de texto.
        return str(file_path)

    def save_processed(self, image, filename: str) -> str:
        """
        Guarda una imagen procesada en la carpeta 'xray_proc'.

        Args:
            image: La imagen procesada, probablemente un array de NumPy (formato de OpenCV).
            filename (str): El nombre del archivo a guardar.

        Returns:
            str: La ruta completa del archivo guardado como una cadena de texto.
        """
        # Combina la ruta de la carpeta 'xray_proc' con el nombre del archivo.
        file_path = self.proc_dir / filename
        # Utiliza cv2.imwrite() para guardar la imagen en el disco.
        # Espera que la ruta sea una cadena, por lo que convertimos el objeto Path.
        cv2.imwrite(str(file_path), image)
        # Devuelve la ruta del archivo guardado.
        return str(file_path)