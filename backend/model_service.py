import threading
from io import BytesIO

import mlflow.pytorch
import torch
from PIL import Image

from alzheimer_classification.models import VGG16Model
from alzheimer_classification.utils.definitions import CLASS_NAMES


class ModelService:
    def __init__(self, model_uri: str):
        self.lock = threading.Lock()
        self.model_uri = model_uri
        self.model = self._load_model()
        self.transform = VGG16Model.get_test_transform()

    def _load_model(self) -> torch.nn.Module:
        return mlflow.pytorch.load_model(self.model_uri)

    def reload(self, new_model_uri: str) -> None:
        with self.lock:
            self.model_uri = new_model_uri
            self.model = self._load_model()

    def predict(self, image_bytes: bytes) -> tuple[str, float]:
        with self.lock:
            image_tensor = self._preprocess_image(image_bytes)
            predictions = self._predict_tensor(image_tensor)

        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = torch.max(predictions).item()
        class_name = CLASS_NAMES[predicted_class]
        return class_name, confidence

    def _preprocess_image(self, image_bytes: bytes) -> torch.Tensor:
        image = Image.open(BytesIO(image_bytes)).convert("L")
        tensor = self.transform(image).unsqueeze(0)
        return tensor

    def _predict_tensor(self, image_tensor: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            output = self.model(image_tensor)
            return torch.nn.functional.softmax(output, dim=1)
