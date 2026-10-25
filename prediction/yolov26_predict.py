from pathlib import Path

from ultralytics import YOLO
from prediction.predict import Predict

class YoloV26Predict(Predict):
    MODEL_PATH = "yolo26n.pt"

    def _load_model(self, base_path: str):
        return YOLO(Path(base_path, self.MODEL_PATH))