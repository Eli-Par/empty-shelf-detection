import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.rtdetr_train import RtDetrTrain
from training.yolov11_train import Yolov11Train
from training.yolov26_train import Yolov26Train
from training.yolov8_train import Yolov8Train

from augmentations.gaussian_noise_augmentation import GaussianNoiseAugmentation
from augmentations.brightness_contrast_augmentation import BrightnessContrastAugmentation
from augmentations.flip_augmentation import FlipAugmentation
from augmentations.standard_augmentor import StandardAugmentor
from augmentations.pass_through_augmentor import PassThroughAugmentor
from test import bulk_trainer


bulk_trainer.train_bulk(
    [
        (
            "No Augmentations",
            lambda: PassThroughAugmentor()
        ),
        (
            "Standard Augmentations",
            lambda: StandardAugmentor([
                FlipAugmentation(),
                BrightnessContrastAugmentation(alpha=1.0, beta=40),
                GaussianNoiseAugmentation(mean=0, std=10),
            ])
        ),
    ],
    [
        # lambda *args: Yolov8Train(*args, epochs=3),
        # lambda *args: Yolov11Train(*args, epochs=3),
        # lambda *args: Yolov26Train(*args, epochs=3),
        # lambda *args: RtDetrTrain(*args, epochs=3),
    ]
)