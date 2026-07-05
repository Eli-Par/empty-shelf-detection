from pathlib import Path
import shutil

import cv2

from training.train import Train
from ultralytics import YOLO
import torch

class YoloTrain(Train):

    def __init__(self, model_path, train_images, train_boxes, validate_images, validate_boxes, test_images, test_boxes, epochs=50, batch_size=16, img_size=640, lr0=0.01):
        self.model_path = model_path
        self.model = YOLO(model_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.img_size = img_size
        self.lr0 = lr0

        self._setup_train_files(train_images, train_boxes, validate_images, validate_boxes, test_images, test_boxes)

    def _setup_train_files(self, train_images, train_boxes, validate_images, validate_boxes, test_images, test_boxes):
        root = Path("temp/yolo")

        if root.exists():
            shutil.rmtree(root)

        root.mkdir(parents=True)

        self._write_data(root, "train", train_images, train_boxes)
        self._write_data(root, "val", validate_images, validate_boxes)
        self._write_data(root, "test", test_images, test_boxes)

        classes_set = set()
        for dataset in [train_boxes, validate_boxes, test_boxes]:
            for boxes in dataset:
                for box in boxes:
                    classes_set.add(box.type)
        classes = sorted(classes_set)

        self.data_yaml = "temp/yolo/data.yaml"
        with open(self.data_yaml, "w") as f:
            f.write("train: ../train/images\n")
            f.write("val: ../val/images\n")
            f.write("test: ../test/images\n\n")

            f.write(f"nc: {len(classes)}\n")
            f.write(f"names: {[str(c) for c in classes]}\n")

    def _write_data(self, root, split_name, images, boxes):
        images_dir = root / split_name / "images"
        labels_dir = root / split_name / "labels"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for index, (image, image_boxes) in enumerate(zip(images, boxes)):
            stem = f"{index:06d}"

            image_path = images_dir / f"{stem}.jpg"
            label_path = labels_dir / f"{stem}.txt"

            cv2.imwrite(image_path, image)

            h, w = image.shape[:2]

            with open(label_path, "w") as f:
                for box in image_boxes:
                    x, y, bw, bh = box.box

                    cx = (x + bw / 2) / w
                    cy = (y + bh / 2) / h
                    norm_w = bw / w
                    norm_h = bh / h

                    f.write(
                        f"{box.type} "
                        f"{cx:.6f} {cy:.6f} "
                        f"{norm_w:.6f} {norm_h:.6f}\n"
                    )

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
            plots=True,

            augment=False,
            mosaic=0.0,
            mixup=0.0,
            fliplr=0.0,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,

            project=self.project if self.project is not None else "",
            name=f"{self.model_path.removesuffix(".pt")}"
        )

        return results