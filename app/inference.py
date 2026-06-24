import io
import logging
from typing import List, Optional

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import settings
from app.model_loader import model_loader
from app.schemas import Detection, PredictionResponse

logger = logging.getLogger(__name__)


TABLE_LABEL_MAPPING = {
    "table_activities": "activity",
    "table_balances": "balance",
}

TABLE_MODEL_CLASSES = set(TABLE_LABEL_MAPPING.keys())


async def read_image(file: UploadFile) -> Image.Image:
    """
    Reads an uploaded file and converts it to an RGB PIL image.
    Raises HTTP 400 if the file is empty or not a valid image.
    """
    content = await file.read()

    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
        return image
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image")


def parse_yolo_results(results) -> List[Detection]:
    """
    Converts raw YOLO results into a clean list of Detection objects.
    """
    detections: List[Detection] = []

    if not results:
        return detections

    result = results[0]
    names = result.names

    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = names[class_id]
        confidence = float(box.conf[0].item())
        bbox = [float(x) for x in box.xyxy[0].tolist()]

        detections.append(
            Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=round(confidence, 4),
                bbox=[round(x, 2) for x in bbox],
            )
        )

    return detections


def map_model_class_to_business_label(class_name: str) -> Optional[str]:
    """
    Maps model-specific class names to the labels required by the assignment.
    """
    return TABLE_LABEL_MAPPING.get(class_name)


def choose_final_table_prediction(detections: List[Detection]) -> Optional[Detection]:
    """
    Selects the final table prediction only from table-related classes.

    The model detects multiple object types:
    - table_activities
    - table_balances
    - columns
    - text areas
    - account opinion
    """
    table_detections = [
        detection
        for detection in detections
        if detection.class_name in TABLE_MODEL_CLASSES
    ]

    if not table_detections:
        return None

    return max(table_detections, key=lambda detection: detection.confidence)


async def predict_image(file: UploadFile) -> PredictionResponse:
    """
    Full prediction flow:
    1. Read uploaded image.
    2. Run YOLO inference.
    3. Parse all detections.
    4. Select the best table detection only.
    5. Return the business label: activity / balance.
    """
    image = await read_image(file)
    model = model_loader.load_model()

    logger.info("Running inference for file: %s", file.filename)

    results = model.predict(
        source=image,
        conf=settings.confidence_threshold,
        imgsz=settings.image_size,
        verbose=False,
    )

    detections = parse_yolo_results(results)
    best_table_detection = choose_final_table_prediction(detections)

    if best_table_detection is None:
        logger.info("No balance/activity table detected for file: %s", file.filename)

        return PredictionResponse(
            filename=file.filename or "unknown",
            prediction=None,
            confidence=None,
            model_class_name=None,
            detections=detections,
            message="No balance/activity table detected",
        )

    business_label = map_model_class_to_business_label(best_table_detection.class_name)

    logger.info(
        "Prediction completed. file=%s prediction=%s model_class=%s confidence=%s",
        file.filename,
        business_label,
        best_table_detection.class_name,
        best_table_detection.confidence,
    )

    return PredictionResponse(
        filename=file.filename or "unknown",
        prediction=business_label,
        confidence=best_table_detection.confidence,
        model_class_name=best_table_detection.class_name,
        detections=detections,
        message="Prediction completed successfully",
    )