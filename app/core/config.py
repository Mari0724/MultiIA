import os
from dotenv import load_dotenv

# Cargar variables del .env
load_dotenv()

class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "MultiIA")
    APP_ENV: str = os.getenv("APP_ENV", "development")
    APP_VERSION: str = os.getenv("APP_VERSION", "1.0.0")
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "True") == "True"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "changeme")

# Instancia global de settings
settings = Settings()
