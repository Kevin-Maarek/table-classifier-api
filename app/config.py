from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Financial Table Classifier API"
    model_path: str = "models/best.pt"
    confidence_threshold: float = 0.25
    image_size: int = 640


settings = Settings()