from abc import ABC, abstractmethod
from cv2 import Mat
import torch

from data_classes import BoundingBox

class Predict(ABC):
    MODEL_ROOT = "trained_models"

    def __init__(self, augmentation_path: str, confidence=0.25, iou=0.7, image_size=640):
        self.confidence = confidence
        self.iou = iou
        self.image_size = image_size

        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(f"{self.MODEL_ROOT}/{augmentation_path}")

    @abstractmethod
    def _load_model(self, augmentation_path: str):
        pass

    def predict(self, image: Mat) -> list[BoundingBox]:
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        return self._convert_results(results)

    def _convert_results(self, results) -> list[BoundingBox]:
        boxes = []

        for result in results:
            if result.boxes is None:
                continue

            xyxy = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()

            for box, obj_type in zip(xyxy, classes):
                x1, y1, x2, y2 = box

                x = int(round(x1))
                y = int(round(y1))
                width = int(round(x2 - x1))
                height = int(round(y2 - y1))

                boxes.append(
                    BoundingBox(
                        obj_type=int(obj_type),
                        box=(x, y, width, height),
                    )
                )

        return boxes