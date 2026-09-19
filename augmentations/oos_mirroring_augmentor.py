# used to tell if a pass actually changed the image
import numpy as np

# the base class we are implementing
from augmentations.interface.augmentor import Augmentor
# the one pass mirroring we reuse for each round
from augmentations.oos_mirroring_augmentation import OosMirroringAugmentation


class OosMirroringAugmentor(Augmentor):
    """
    The more faithful paper version of the out of stock mirroring (N chained rounds).
    Unlike the plain augmentation, which does one pass and one image, this makes
    N new images per original by chaining: round 2 mirrors round 1's output, not
    the original, so each image is a little more emptied than the last.

    The paper uses N = 2, so one original turns into two new images.
    Only images that have an empty gap (class 0) get expanded. Images with no
    empty gap are carried through unchanged, like the originals.

    This is an Augmentor (works over the whole dataset) not an Augmentation,
    because one image in makes several images out, which the one in one out
    Augmentation contract cannot express. It does not go inside StandardAugmentor.

    How to use:
        Give it the loaded dataset and it returns the originals plus the
        chained copies. iterations is the paper's N. shrink_factor and seed
        are passed straight through to the underlying one pass mirroring.

        expander = OosMirroringAugmentor()
        aug_images, aug_boxes = expander.augment(images, boxes)
    """

    def __init__(self, iterations=2, shrink_factor=0.75, seed=None):
        # how many chained rounds per image (paper's N)
        self.iterations = iterations
        # the one pass mirroring we call each round
        self.augmentation = OosMirroringAugmentation(shrink_factor=shrink_factor, seed=seed)

    def augment(self, images, bounding_boxes):
        # start with the original images
        result_images = list(images)
        # start with the original box lists
        result_boxes = list(bounding_boxes)

        # go through each original image with its boxes
        for index in range(len(images)):
            image = images[index]
            boxes = bounding_boxes[index]
            # each round builds on the previous round's output, not the original
            cur_image = image
            cur_boxes = boxes
            # do N chained rounds
            for _ in range(self.iterations):
                # one mirroring pass
                result = self.augmentation.augment(cur_image, cur_boxes)
                new_image = result[0]
                new_boxes = result[1]
                # nothing left to grow (no gap, or no room) so stop chaining
                if np.array_equal(new_image, cur_image):
                    break
                # add this rounds image and boxes to the output
                result_images.append(new_image)
                result_boxes.append(new_boxes)
                # chain, the next round grows this result further
                cur_image = new_image
                cur_boxes = new_boxes

        # return originals plus all chained copies
        return result_images, result_boxes
