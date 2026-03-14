import cv2

from augmentations.interface.augmentation import Augmentation

class FlipAugmentation(Augmentation):
    
    def augment(self, image):
        return cv2.flip(image, 1)