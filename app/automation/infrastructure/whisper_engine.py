
import os

# 🔹 Asegurar que ffmpeg esté en el PATH (Windows)
os.environ["PATH"] += os.pathsep + r"C:\Users\USER\ffmpeg-8.0-essentials_build\bin"


import whisper
from app.automation.domain.automation_interface import VoiceToTextInterface

class WhisperEngine(VoiceToTextInterface):
    def __init__(self, model_size: str = "base"):
        """
        model_size puede ser: tiny, base, small, medium, large
        - tiny/base → más rápido, menos preciso
        - medium/large → más preciso, más pesado
        """
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path, language="es")
        return result["text"]


"""
Implementación concreta de la interfaz `VoiceToTextInterface` usando el modelo Whisper.

- Esta clase cumple el contrato definido en la interfaz: debe tener un método `transcribe`
  que reciba la ruta de un archivo de audio y devuelva texto.
- Aquí sí damos la implementación real: cargamos un modelo Whisper y usamos su función
  `transcribe`.

Detalles:
- En el constructor (`__init__`) se carga el modelo Whisper con un tamaño elegido:
    * "tiny" o "base" → más rápidos, pero menos precisos.
    * "small", "medium", "large" → más precisos, pero más pesados y lentos.
- El modelo queda guardado en `self.model` para reutilizarlo sin volver a cargarlo.

- En `transcribe(audio_path: str)`:
    * Se pasa el archivo de audio a `self.model.transcribe`.
    * Se indica `language="es"` para procesar directamente en español.
    * El método devuelve solo el texto final (`result["text"]`).

Ventaja:
- Como esta clase implementa la interfaz `VoiceToTextInterface`, puede reemplazarse por
  otro motor (Google Speech, Azure, etc.) sin modificar la lógica de negocio. Solo se cambia
  esta implementación, el resto del sistema ni se entera.
"""
