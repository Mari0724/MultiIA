import subprocess
from pathlib import Path

# Ruta absoluta a tu ffmpeg
ffmpeg_path = r"C:\Users\USER\ffmpeg-8.0-essentials_build\bin\ffmpeg.exe"

def convert_to_wav(input_file: str, output_file: str):
    """
    Convierte un archivo de audio (mp3, m4a, etc.) a wav usando ffmpeg.
    """
    if not Path(input_file).exists():
        raise FileNotFoundError(f"❌ No se encontró el archivo de entrada: {input_file}")

    subprocess.run([ffmpeg_path, "-i", input_file, output_file], check=True)
    print(f"✅ Conversión lista: {output_file}")


if __name__ == "__main__":
    # Ejemplo de uso
    convert_to_wav("audio.mp3", "audio.wav")
