from pathlib import Path

from ultralytics import RTDETR
from prediction.predict import Predict

class RtDetrPredict(Predict):
    MODEL_PATH = "rt_detr.pt"

    def _load_model(self, base_path: str):
        return RTDETR(Path(base_path, self.MODEL_PATH))