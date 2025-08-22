# run.py
import uvicorn
from app.core.config import settings  # tu config centralizada

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,   # ← lo sacamos de .env
        port=int(settings.PORT),  # ← también de .env
        reload=True
    )
