import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from training.yolov8_train import Yolov8Train


trainer = Yolov8Train(
    model_path="yolov8n.pt",
    data_yaml="data/data.yaml",
    epochs=50,
    batch_size=16,
    img_size=640
)

trainer.train()