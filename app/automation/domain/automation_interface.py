from typing import Protocol

class VoiceToTextInterface(Protocol):
    def transcribe(self, audio_path: str) -> str:
        """Transcribe un archivo de audio y devuelve texto"""
        ...


"""
Esta clase define una INTERFAZ usando `Protocol`, que funciona como un contrato.
- `Protocol` en Python → permite declarar métodos que otras clases deben implementar,
  parecido a las interfaces en Java o C#.
- Aquí declaramos el método `transcribe(audio_path: str) -> str`, que recibe la ruta de un
  archivo de audio y devuelve el texto transcrito.

El `...` (ellipsis) significa:
- "Aquí no pongo implementación, solo marco que este método existe".
- Es equivalente a un `pass`, pero se usa mucho en interfaces para dejar claro que
  este espacio NO se llena aquí, sino en la clase que implemente la interfaz.
- Sirve para que cualquier motor (Whisper, Google Speech, Azure, etc.) sepa qué función
  debe tener y qué debe devolver, sin importar cómo lo haga internamente.

Ventaja de esta separación:
- La lógica de negocio (nuestro sistema necesita convertir voz a texto) no depende
  de un motor específico.
- La implementación (cómo se logra esa conversión) puede cambiar libremente.
  Hoy usamos Whisper, mañana podríamos usar Google Speech, y no habría que modificar
  el resto del código, porque todos cumplen el mismo contrato (`transcribe`).
"""
