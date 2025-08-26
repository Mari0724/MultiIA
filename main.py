from fastapi import FastAPI
from app.core.config import settings   # config centralizada

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Proyecto SENASOFT 2025 con IA, Python y FastAPI 🚀",
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Rutas globales
    @app.get("/")
    def read_root():
        return {
            "message": f"Bienvenida a {settings.APP_NAME} 🚀",
            "env": settings.APP_ENV
        }

    # Importa routers de cada módulo IA
    from app.nlp.api.routes import router as nlp_router
    from app.vision.api.routes import router as vision_router
    from app.recomendation.api.routes import router as recommendation_router
    from app.prediction.api.routes import router as prediction_router

    # Incluye routers
    app.include_router(nlp_router, prefix="/nlp", tags=["NLP"])
    app.include_router(vision_router, prefix="/vision", tags=["Visión"])
    app.include_router(recommendation_router, prefix="/recommendation", tags=["Recomendación"])
    app.include_router(prediction_router)
    return app

app = create_app()
