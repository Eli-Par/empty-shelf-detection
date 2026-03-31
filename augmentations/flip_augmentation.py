import cv2

from augmentations.interface.augmentation import Augmentation
from data_classes import BoundingBox

class FlipAugmentation(Augmentation):
    
    def augment(self, image, bounding_boxes):
        flipped_image = cv2.flip(image, 1)

        flipped_boxes = []
        h, w = image.shape[:2]
        for bbox in bounding_boxes:
            x, y, bw, bh = bbox.box

            flipped_boxes.append(BoundingBox(bbox.type, (w - x + bw, y, bw, bh)))

        return flipped_image, flipped_boxes