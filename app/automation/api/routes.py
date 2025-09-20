from fastapi import APIRouter, UploadFile, File
import shutil
import os
from app.automation.application.automation_service import AutomationService

router = APIRouter(prefix="/automation", tags=["Automation"])
service = AutomationService()

@router.post("/voice-to-text")
async def voice_to_text(file: UploadFile = File(...)):
    # Guardar audio temporal
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Transcribir con Whisper
    text = service.voice_to_text(temp_path)

    # Eliminar temporal
    os.remove(temp_path)

    return {"transcription": text}


"""
Este archivo define las rutas (endpoints) relacionadas con el módulo de Automatización,
en este caso la funcionalidad de **Voz a Texto**.

Rol dentro de la arquitectura:
- Pertenece a la capa API, la cual expone los servicios a través de HTTP.
- Usa FastAPI para definir un router (`APIRouter`) con prefijo `/automation`
  y etiqueta "Automation" para que se agrupe bonito en Swagger.

Flujo del endpoint `/voice-to-text`:
1. Recibe un archivo de audio (`UploadFile`) vía POST.
2. Guarda temporalmente el archivo en disco con un nombre dinámico `temp_<nombre>`.
   - Se usa `shutil.copyfileobj` para copiar el contenido del archivo subido al temporal.
3. Llama al servicio de aplicación (`AutomationService`) para transcribir el audio.
   - Aquí no sabemos si es Whisper u otro motor, porque el service se encarga de esa lógica.
4. Borra el archivo temporal (`os.remove`) para no dejar basura en disco.
5. Devuelve un JSON con la transcripción en la clave `"transcription"`.

Ventaja de este diseño:
- La API no tiene que saber nada de IA ni de cómo funciona Whisper.
- Solo orquesta: recibe archivo → guarda temporal → pide al servicio la transcripción → responde.
- Esto mantiene la separación de responsabilidades: 
  la API maneja HTTP, el servicio maneja negocio, y la infraestructura maneja la implementación técnica.
"""
