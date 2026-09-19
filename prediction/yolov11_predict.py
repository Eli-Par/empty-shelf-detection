from pathlib import Path

from ultralytics import YOLO
from prediction.predict import Predict


class YoloV8Predict(Predict):
    MODEL_PATH = "yolo11n.pt"

    def _load_model(self, base_path: str):
        return YOLO(Path(base_path, self.MODEL_PATH))