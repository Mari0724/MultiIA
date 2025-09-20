from app.automation.infrastructure.whisper_engine import WhisperEngine

class AutomationService:
    def __init__(self):
        self.engine = WhisperEngine(model_size="base")  # 👈 puedes cambiarlo

    def voice_to_text(self, audio_path: str) -> str:
        return self.engine.transcribe(audio_path)


"""
Este archivo contiene la lógica de aplicación, representada en la clase `AutomationService`.

Rol dentro de la arquitectura:
- Actúa como intermediario entre la API (controladores/rutas) y la infraestructura (motores de IA).
- Su objetivo es ofrecer un servicio limpio y fácil de usar que la API pueda invocar sin
  preocuparse por detalles técnicos.

Detalles:
- En el constructor (`__init__`) inicializamos `WhisperEngine` indicando el tamaño del modelo
  (por defecto "base", pero puede cambiarse a "tiny", "small", "medium", etc.).
- El método `voice_to_text(audio_path: str)` recibe la ruta de un archivo de audio y delega
  la transcripción al motor (`self.engine.transcribe`).

Ventaja de esta capa:
- La API nunca habla directamente con Whisper ni con ninguna librería externa.
- Si mañana cambiamos Whisper por Google Speech o Azure Speech, solo se modifica
  `WhisperEngine`. El `AutomationService` mantiene la misma interfaz para la API,
  garantizando independencia entre la lógica de aplicación y la implementación concreta.
"""
