import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.yolov26_train import Yolov26Train
from training.yolov11_train import Yolov11Train

from augmentations.gaussian_noise_augmentation import GaussianNoiseAugmentation
from augmentations.brightness_contrast_augmentation import BrightnessContrastAugmentation
from augmentations.flip_augmentation import FlipAugmentation
from augmentations.standard_augmentor import StandardAugmentor
from data_loader.loader import Loader
from training.yolov8_train import Yolov8Train


loader = Loader("data/empty-shelf-data/data.yaml")

print("Loader loaded")

train_images, train_boxes = loader.load("train")
print(f"{len(train_images)} train images loaded")

validate_images, validate_boxes = loader.load("val")
print(f"{len(validate_images)} validation images loaded")


test_images, test_boxes = loader.load("test")
print(f"{len(test_images)} test images loaded")

augmentor = StandardAugmentor([
    FlipAugmentation(),
    BrightnessContrastAugmentation(alpha=1.0, beta=40),
    GaussianNoiseAugmentation(mean=0, std=10),
])
aug_images, aug_boxes = augmentor.augment(train_images, train_boxes)

models_base = [
    Yolov8Train(train_images, train_boxes, validate_images, validate_boxes, test_images, test_boxes),
    Yolov11Train(train_images, train_boxes, validate_images, validate_boxes, test_images, test_boxes),
    Yolov26Train(train_images, train_boxes, validate_images, validate_boxes, test_images, test_boxes)
]

models_aug = [
    Yolov8Train(aug_images, aug_boxes, validate_images, validate_boxes, test_images, test_boxes),
    Yolov11Train(aug_images, aug_boxes, validate_images, validate_boxes, test_images, test_boxes),
    Yolov26Train(aug_images, aug_boxes, validate_images, validate_boxes, test_images, test_boxes)
]

for model in models_base:
    model.setProject("no_augmentations")
    print("model setup finished")

    model.train()

for model in models_aug:
    model.setProject("standard_augmentations")
    print("model setup finished")

    model.train()