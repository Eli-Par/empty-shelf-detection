# the base class we are implementing
from augmentations.interface.augmentor import Augmentor
# the type used in the constructor
from augmentations.interface.augmentation import Augmentation


class StandardAugmentor(Augmentor):
    """
    Grows the dataset by running augmentations on it.
    Each augmentation makes one new copy of every image,
    so the output is the originals plus all the copies.
    Copies are made from the original, not stacked.

    How to use:
        Give it a list of augmentations:

        augmentor = StandardAugmentor([
            FlipAugmentation(),
            BrightnessContrastAugmentation(alpha=1.0, beta=40),
            GaussianNoiseAugmentation(mean=0, std=10),
        ])
        aug_images, aug_boxes = augmentor.augment(images, boxes)
    """

    def __init__(self, augmentations: list[Augmentation]):
        # store the augmentations to apply
        self.augmentations = augmentations

    def augment(self, images, bounding_boxes):
        # start with the original images
        result_images = list(images)
        # start with the original box lists
        result_boxes = list(bounding_boxes)

        # go through each augmentation
        for augmentation in self.augmentations:
            # apply it to every original image
            for image, boxes in zip(images, bounding_boxes):
                # make one augmented copy
                aug_image, aug_boxes = augmentation.augment(image, boxes)
                # add the augmented image
                result_images.append(aug_image)
                # add the augmented boxes
                result_boxes.append(aug_boxes)

        # return originals plus all copies
        return result_images, result_boxes
