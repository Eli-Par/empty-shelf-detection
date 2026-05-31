# used to flip the image
import cv2

# the base class we are implementing
from augmentations.interface.augmentation import Augmentation
# our bounding box data class
from data_classes import BoundingBox


class FlipAugmentation(Augmentation):
    """
    Flips the image left to right like a mirror.
    Moves the boxes too so they still match.

    How to use:
        Takes no parameters.

        flip = FlipAugmentation()
    """

    def augment(self, image, bounding_boxes):
        # flip the image sideways
        flipped_image = cv2.flip(image, 1)

        # will hold the moved boxes
        flipped_boxes = []
        # image height and width
        h, w = image.shape[:2]
        # go through each box
        for bbox in bounding_boxes:
            # pull out the box values
            x, y, bw, bh = bbox.box

            # mirror the x and keep the rest
            flipped_boxes.append(BoundingBox(bbox.type, (w - x - bw, y, bw, bh)))

        # return the flipped image and boxes
        return flipped_image, flipped_boxes
