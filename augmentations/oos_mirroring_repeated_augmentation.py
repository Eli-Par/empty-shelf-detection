import numpy as np

from augmentations.interface.augmentation import Augmentation
from augmentations.oos_mirroring_augmentation import OosMirroringAugmentation


class OosMirroringRepeatedAugmentation(Augmentation):

    def __init__(self, iterations = 2, shrink_factor=0.75, seed=None):
        self.iterations = iterations
        self.augmentation = OosMirroringAugmentation(shrink_factor=shrink_factor, seed=seed)
    
    def augment(self, image, bounding_boxes):
        curr_image = image.copy()
        curr_bounding_boxes = list(bounding_boxes)

        for _ in range(self.iterations):
            new_image, new_bounding_boxes = self.augmentation.augment(new_image, new_bounding_boxes)

            if np.array_equal(new_image, curr_image):
                break

            curr_image = new_image
            curr_bounding_boxes = new_bounding_boxes

        return curr_image, curr_bounding_boxes