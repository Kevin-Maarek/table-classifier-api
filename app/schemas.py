from typing import List, Optional

from pydantic import BaseModel


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]


class PredictionResponse(BaseModel):
    filename: str
    prediction: Optional[str]
    confidence: Optional[float]
    model_class_name: Optional[str]
    detections: List[Detection]
    message: str


class HealthResponse(BaseModel):
    status: str
    service: str


class ModelInfoResponse(BaseModel):
    model_path: str
    task: str
    class_names: dict
    num_classes: int
    num_parameters: Optional[int]
    device: Optional[str]
