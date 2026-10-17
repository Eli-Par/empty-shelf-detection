import random

import cv2

from augmentations.interface.augmentation import Augmentation
from augmentations.move_augmention import MoveAugmentation
from data_classes import BoundingBox


class MoveMultiAugmentation(Augmentation):

    def __init__(self, move_percent_min=0.25, move_percent_max=1,  full_to_partial_rate=0.75, max_move_dist=0.75, min_height=0.25, max_height=0.6, max_tries=10000, seed=100):
        self.move_percent_min = move_percent_min
        self.move_percent_max = move_percent_max
        self.full_to_partial_rate = full_to_partial_rate
        self.max_move_dist = max_move_dist
        self.min_height = min_height
        self.max_height = max_height
        self.max_tries = max_tries

        self.rng = random.Random(seed)

        self.moveAugmentation = MoveAugmentation(full_to_partial_rate, max_move_dist, min_height, max_height, max_tries, self.rng.randint(-(2**31), 2**31 - 1))

    def augment(self, image: cv2.Mat, bounding_boxes: list[BoundingBox]):
        new_image = image
        new_bounding_boxes = bounding_boxes

        repetitions = self.rng.randint(max(round(len(bounding_boxes) * self.move_percent_min), 1), round(len(bounding_boxes) * self.move_percent_max))
        for _ in range(repetitions):
            new_image, new_bounding_boxes = self.moveAugmentation.augment(new_image, new_bounding_boxes)

        return new_image, new_bounding_boxes