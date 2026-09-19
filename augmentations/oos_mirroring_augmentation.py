# used to flip the copied pixels
import cv2
# used to pick a random gap and a random direction
import random

# the base class we are implementing
from augmentations.interface.augmentation import Augmentation
# our bounding box data class
from data_classes import BoundingBox

# class id for an empty (out of stock) gap
OOS_TYPE = 0


class OosMirroringAugmentation(Augmentation):
    """
    Recreates the out of stock mirroring trick from the paper.
    Takes an empty shelf gap, copies its pixels, flips them sideways,
    and pastes them into the slot right next to it so the gap grows
    wider by about one product slot. The gap's box is widened to match.

    Only empty gaps (class 0) are grown. Front gaps (class 1) are left alone.
    Does one mirroring pass and gives back one image, so it drops into
    StandardAugmentor just like the other augmentations.

    How to use:
        Just create it and add it to your augmentation list:

        oos = OosMirroringAugmentation()

        Two optional settings:
        - shrink_factor: if the pasted area would run into another gap,
          it gets made smaller by this much each time (0.75 = shrink to
          three quarters) until it fits. The default is fine.
        - seed: set a number to make the random choices repeat the same
          way every run. Leave it out normally.
    """

    def __init__(self, shrink_factor=0.75, seed=None):
        # how much to shrink the paste region on overlap (paper's T)
        self.shrink_factor = shrink_factor
        # own random generator so a seed makes results repeatable
        self.rng = random.Random(seed)

    def augment(self, image, bounding_boxes):
        # image height and width
        h = image.shape[0]
        w = image.shape[1]

        # find the empty gaps, these are the only ones we grow
        oos_indices = []
        # go through each box and keep the empty ones
        for i in range(len(bounding_boxes)):
            if bounding_boxes[i].type == OOS_TYPE:
                oos_indices.append(i)

        # no empty gap to grow, hand the image back unchanged
        if not oos_indices:
            return image, bounding_boxes

        # pick one empty gap at random to grow
        chosen = self.rng.choice(oos_indices)
        o = bounding_boxes[chosen]
        # pull out the gap box values
        ox = o.box[0]
        oy = o.box[1]
        ow = o.box[2]
        oh = o.box[3]
        # left and right edges of the gap
        ox1 = ox
        ox2 = ox + ow

        # center x of the gap, used to pick a direction
        cx = ox1 + ow / 2

        # decide which way to grow (paper equation 1)
        if cx < 0.25 * w:
            # gap hugs the left edge, grow to the right into the image
            direction = "right"
        elif cx > 0.75 * w:
            # gap hugs the right edge, grow to the left into the image
            direction = "left"
        else:
            # gap is in the middle, either side is fine
            direction = self.rng.choice(["left", "right"])

        # build the paste region next to the gap, same height as the gap
        if direction == "right":
            # paste region sits to the right of the gap
            ec_x1 = ox2
            # start one gap width wide, clamped to the image edge
            ec_x2 = min(ox2 + ow, w)
        else:
            # paste region sits to the left of the gap
            ec_x2 = ox1
            # start one gap width wide, clamped to the image edge
            ec_x1 = max(ox1 - ow, 0)

        # shrink the paste region until it clears every other empty gap
        while self._overlaps_other_gap(ec_x1, ec_x2, oy, oy + oh, bounding_boxes, chosen):
            # current width of the paste region
            ec_w = ec_x2 - ec_x1
            # shrink it by the shrink factor
            new_w = int(ec_w * self.shrink_factor)
            # always shrink by at least one pixel so the loop can end
            if new_w >= ec_w:
                new_w = ec_w - 1
            # no room left to grow into, give the image back unchanged
            if new_w <= 1:
                return image, bounding_boxes
            # keep the edge touching the gap fixed, pull in the outer edge
            if direction == "right":
                ec_x2 = ec_x1 + new_w
            else:
                ec_x1 = ec_x2 - new_w

        # copy on a fresh image so we never touch the original pixels
        new_image = image.copy()
        # the gap's own pixels
        patch = new_image[oy:oy + oh, ox1:ox2]
        # flip them sideways like a mirror
        flipped = cv2.flip(patch, 1)
        # how wide the paste region is (may be less than the gap after shrinking)
        target_w = ec_x2 - ec_x1
        # take the flipped columns nearest the gap
        if direction == "right":
            piece = flipped[:, :target_w]
        else:
            piece = flipped[:, flipped.shape[1] - target_w:]
        # paste the mirrored pixels into the paste region
        new_image[oy:oy + oh, ec_x1:ec_x1 + piece.shape[1]] = piece

        # widen the gap box to cover the gap plus the new empty area
        nx1 = min(ox1, ec_x1)
        nx2 = max(ox2, ec_x2)
        grown = BoundingBox(o.type, (nx1, oy, nx2 - nx1, oh))

        # rebuild the box list, swapping the chosen gap for its grown version
        new_boxes = []
        # go through each box
        for i in range(len(bounding_boxes)):
            b = bounding_boxes[i]
            if i == chosen:
                # the grown gap replaces the original
                new_boxes.append(grown)
            else:
                # every other box is carried over unchanged
                new_boxes.append(BoundingBox(b.type, b.box))

        # return the mirrored image and updated boxes
        return new_image, new_boxes

    def _overlaps_other_gap(self, ec_x1, ec_x2, ec_y1, ec_y2, bounding_boxes, chosen):
        # check the paste region against every other empty gap
        for i in range(len(bounding_boxes)):
            b = bounding_boxes[i]
            # skip the gap we are growing and any non empty box
            if i == chosen or b.type != OOS_TYPE:
                continue
            # pull out the other box values
            bx = b.box[0]
            by = b.box[1]
            bw = b.box[2]
            bh = b.box[3]
            # its edges
            bx1 = bx
            bx2 = bx + bw
            by1 = by
            by2 = by + bh
            # standard rectangle overlap test
            if ec_x1 < bx2 and bx1 < ec_x2 and ec_y1 < by2 and by1 < ec_y2:
                return True
        # no overlap with any other gap
        return False
