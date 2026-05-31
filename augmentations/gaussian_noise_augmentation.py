# used to make noise and clip the array
import numpy as np

# the base class we are implementing
from augmentations.interface.augmentation import Augmentation


class GaussianNoiseAugmentation(Augmentation):
    """
    Adds random grain to the image like a grainy photo.
    Boxes stay the same.

    How to use:
        mean is the center of the noise, keep at 0
        std is how strong the grain is, higher is more, try 5 to 25

        light_grain = GaussianNoiseAugmentation(mean=0, std=10)
        heavy_grain = GaussianNoiseAugmentation(mean=0, std=25)
    """

    def __init__(self, mean: float, std: float):
        # center of the noise
        self.mean = mean
        # strength of the noise
        self.std = std

    def augment(self, image, bounding_boxes):
        # make noise the same size as the image
        noise = np.random.normal(self.mean, self.std, image.shape)
        # add noise as float then clip and go back to uint8
        noisy_image = np.clip(image.astype(np.float64) + noise, 0, 255).astype(np.uint8)
        # boxes stay the same since nothing moves
        return noisy_image, bounding_boxes
