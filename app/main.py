import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile

from app.config import settings
from app.inference import predict_image
from app.model_loader import model_loader
from app.schemas import HealthResponse, ModelInfoResponse, PredictionResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup/shutdown lifecycle.

    The model is loaded during startup so the first prediction request
    does not need to load it from disk.
    """
    model_loader.load_model()
    yield


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "ok",
        "service": settings.app_name,
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    model = model_loader.load_model()

    num_parameters = None
    device = None

    try:
        num_parameters = sum(p.numel() for p in model.model.parameters())
    except Exception:
        pass

    try:
        device = str(next(model.model.parameters()).device)
    except Exception:
        pass

    return {
        "model_path": settings.model_path,
        "task": model.task,
        "class_names": model.names,
        "num_classes": len(model.names),
        "num_parameters": num_parameters,
        "device": device,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    return await predict_image(file)