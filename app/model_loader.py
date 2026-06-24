import logging
from typing import Optional

from ultralytics import YOLO

from app.config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self):
        self.model: Optional[YOLO] = None

    def load_model(self) -> YOLO:
        """
        Loads the YOLO model once and reuses it for all requests.
        This avoids loading the model from disk on every prediction request.
        """
        if self.model is None:
            logger.info("Loading YOLO model from path: %s", settings.model_path)
            self.model = YOLO(settings.model_path)
            logger.info("Model loaded successfully. Classes: %s", self.model.names)

        return self.model


model_loader = ModelLoader()