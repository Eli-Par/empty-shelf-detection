from abc import ABC, abstractmethod

from cv2 import Mat

from data_classes import BoundingBox

class Augmentation(ABC):
    
    @abstractmethod
    def augment(self, image: Mat, bounding_boxes: list[BoundingBox]):
        pass