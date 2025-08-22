from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def read_root():
    return {"message": "Bienvenida a NeoAI 🚀"}

@router.get("/ping")
def ping():
    return {"pong": True}
