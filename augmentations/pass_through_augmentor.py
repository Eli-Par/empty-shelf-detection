from augmentations.interface.augmentor import Augmentor


class PassThroughAugmentor(Augmentor):
    """
    Returns the images and bounding boxes provided with no augmentations
    """
    def __init__(self):
        pass

    def augment(self, images, bounding_boxes):
        return images, bounding_boxes