from typing import Callable
from cv2 import Mat
from augmentations.interface.augmentor import Augmentor
from data_classes import BoundingBox
from data_loader.loader import Loader
from training.train import Train


def train_bulk(
    augmentor_factories: list[tuple[str, Callable[[], Augmentor]]],
    train_factories: Callable[
        [
            list[Mat],
            list[list[BoundingBox]],
            list[Mat],
            list[list[BoundingBox]],
            list[Mat],
            list[list[BoundingBox]],
        ],
        Train,
    ],
):
    loader = Loader("data/empty-shelf-data/data.yaml")

    print("Loader loaded")

    train_images, train_boxes = loader.load("train")
    print(f"{len(train_images)} train images loaded")

    validate_images, validate_boxes = loader.load("val")
    print(f"{len(validate_images)} validation images loaded")


    test_images, test_boxes = loader.load("test")
    print(f"{len(test_images)} test images loaded")

    for augmentor_factory in augmentor_factories:
        augmentor_name = augmentor_factory[0]
        augmentor = augmentor_factory[1]()

        aug_images, aug_boxes = augmentor.augment(train_images, train_boxes)

        for train_factory in train_factories:
            model = train_factory(aug_images, aug_boxes, validate_images, validate_boxes, test_images, test_boxes)
            model.setProject(augmentor_name)
            print("model setup finished")

            model.train()