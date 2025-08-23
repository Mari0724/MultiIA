from fastapi import APIRouter, HTTPException
from app.prediction.application.prediction_service import (
    train_linear_model, predict_linear,
    train_logistic_model, predict_logistic
)

router = APIRouter()

@router.get("/")
def root():
    return {"message": "Prediction API lista 🚀"}
    
@router.get("/linear/train")
def train_linear():
    return train_linear_model()

@router.get("/linear/predict")
def predict_linear_endpoint(x: float):
    try:
        return predict_linear(x)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Modelo no entrenado")



# --- LOGÍSTICA ---
@router.get("/logistic/train")
def train_logistic():
    return train_logistic_model()

@router.get("/logistic/predict")
def predict_logistic_endpoint(x1: float, x2: float):
    try:
        return predict_logistic(x1, x2)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Modelo no entrenado")
