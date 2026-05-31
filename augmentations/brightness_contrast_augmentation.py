# used to adjust brightness and contrast
import cv2

# the base class we are implementing
from augmentations.interface.augmentation import Augmentation


class BrightnessContrastAugmentation(Augmentation):
    """
    Makes the image lighter, darker, or more punchy.
    Copies different store lighting. Boxes stay the same.

    How to use:
        alpha is contrast, 1.0 is normal, try 0.8 to 1.3
        beta is brightness, 0 is normal, plus is brighter, minus is darker, try -40 to 40

        brighter = BrightnessContrastAugmentation(alpha=1.0, beta=40)
        darker   = BrightnessContrastAugmentation(alpha=1.0, beta=-40)
        punchy   = BrightnessContrastAugmentation(alpha=1.3, beta=0)
    """

    def __init__(self, alpha: float, beta: int):
        # contrast multiplier
        self.alpha = alpha
        # brightness offset
        self.beta = beta

    def augment(self, image, bounding_boxes):
        # apply pixel * alpha + beta and clip to valid range
        adjusted_image = cv2.convertScaleAbs(image, alpha=self.alpha, beta=self.beta)
        # boxes stay the same since nothing moves
        return adjusted_image, bounding_boxes
