from training.train import Train
from ultralytics import YOLO
import torch

class Yolov8Train(Train):

    def __init__(self, model_path, data_yaml, epochs=50, batch_size=16, img_size=640, lr0=0.01):
        self.model = YOLO(model_path)
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.lr0 = lr0

    def train(self):
        device = "0" if torch.cuda.is_available() else "cpu"

        results = self.model.train(
            data=self.data_yaml,
            epochs=self.epochs,
            batch=self.batch_size,
            imgsz=self.img_size,
            lr0=self.lr0,
            patience=10,
            device=device,
            save=True,
            plots=True
        )

        return results