from abc import ABC, abstractmethod

from cv2 import Mat

from data_classes import BoundingBox

class Augmentor(ABC):
    
    @abstractmethod
    def augment(self, images: list[Mat], bounding_boxes: list[list[BoundingBox]]):
        pass