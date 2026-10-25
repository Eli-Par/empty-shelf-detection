# ==========================================================================
# LIGHTING SWAP AUGMENTATION
# ==========================================================================
#
# WHAT IT DOES:
# It takes one empty spot on the shelf and the product sitting right next
# to it, and swaps how bright they are:
#   - the empty spot is made brighter, so it is as bright as the product was
#   - the product is made darker, so it is as dark as the empty spot was
#
# THE STEPS:
#   1. Find all the empty spots in the photo.
#   2. Pick one at random.
#   3. Look at the area right beside it, on the same shelf and the same
#      size. If that area has no label on it, it is product.
#   4. Measure how bright the empty spot is and how bright the product is.
#   5. If the empty spot is not clearly darker than the product, skip it.
#   6. Brighten the empty spot up to the product's brightness.
#   7. Darken the product down to the empty spot's brightness.
#   8. Fade at the edges so there is no hard outline.
#   9. If the chosen spot will not work, try the next spot. If no spot
#      works, hand the photo back unchanged.
#
# THE WEAKNESS IT GOES AFTER:
# The models seem to have a shortcut. In the photos, empty
# spots are almost always dark. So the models know
# "dark rectangle on a shelf means empty", instead of learning what an
# empty shelf actually looks like.
#
# This augmentation breaks the shortcut:
#   - a bright spot can still be empty
#   - a dark spot can still be full of product
# So the model has to look at what is really there, like the back wall of
# the shelf or the edges of products, instead of just how dark it is.
#
# HOW TO USE:
#   One photo in, one photo out, so it goes inside StandardAugmentor like
#   flip, brightness and noise do.
#
#   - max_gain       : the most a spot can be brightened or darkened, so
#                      3.0 means at most 3 times brighter or 3 times darker
#   - min_difference : how much darker the empty spot must be than the
#                      product before we bother swapping (0 to 255 scale)
#   - edge           : how many pixels the soft fade at the edges takes
#   - seed           : set a number to get the same result every run
#
#   swap = LightingSwapGarionAug()
#   swap = LightingSwapGarionAug(max_gain=3.0, min_difference=15, edge=3, seed=0)
# ==========================================================================

# used to blur the fade at the edges and to turn the photo grey
import cv2
# used to do the brightness maths on the pixels
import numpy as np
# used to pick random spots and random sides
import random

# the base class we are building on
from augmentations.interface.augmentation import Augmentation
# our bounding box data class
from data_classes import BoundingBox

# class id for a fully empty spot
EMPTY_TYPE = 0


# swaps the brightness of an empty spot and the product beside it
class LightingSwapGarionAug(Augmentation):

    # sets up the augmentation with its settings
    def __init__(self, max_gain=3.0, min_difference=15, edge=3, seed=None):
        # the most a spot can be brightened or darkened
        self.max_gain = max_gain
        # how much darker the empty spot must be before we swap
        self.min_difference = min_difference
        # how many pixels the soft fade at the edges takes
        self.edge = edge
        # our own random picker so a seed gives the same result every run
        self.rng = random.Random(seed)

    # the main step, called once for each photo
    def augment(self, image, bounding_boxes):
        # will hold the list positions of the empty spots
        empty_indices = []

        # walk through every box we were given
        for i in range(len(bounding_boxes)):
            # this box is an empty spot
            if bounding_boxes[i].type == EMPTY_TYPE:
                # remember where it is in the list
                empty_indices.append(i)

        # there are no empty spots in this photo
        if len(empty_indices) == 0:
            # hand the photo back exactly as it came in
            return image, bounding_boxes

        # mix up the order so the spots get tried in a random order
        self.rng.shuffle(empty_indices)

        # try the empty spots one at a time
        for chosen in empty_indices:
            # try to swap the lighting for this spot
            result = self._try_spot(image, bounding_boxes, chosen)
            # this spot worked
            if result is not None:
                # hand back the new photo and the boxes
                return result

        # none of the spots could be used
        return image, bounding_boxes

    # tries to swap the lighting for one spot, gives back None if it cannot
    def _try_spot(self, image, bounding_boxes, chosen):
        # how tall the whole photo is
        h = image.shape[0]
        # how wide the whole photo is
        w = image.shape[1]

        # grab the box of the spot we are trying
        spot = bounding_boxes[chosen]

        # left edge of the spot, kept inside the photo
        spot_x1 = max(int(spot.box[0]), 0)
        # top edge of the spot, kept inside the photo
        spot_y1 = max(int(spot.box[1]), 0)
        # right edge of the spot, kept inside the photo
        spot_x2 = min(int(spot.box[0] + spot.box[2]), w)
        # bottom edge of the spot, kept inside the photo
        spot_y2 = min(int(spot.box[1] + spot.box[3]), h)

        # how wide the spot is
        spot_w = spot_x2 - spot_x1
        # how tall the spot is
        spot_h = spot_y2 - spot_y1

        # a spot with no width cannot be used
        if spot_w <= 0:
            # tell the caller this spot did not work
            return None

        # a spot with no height cannot be used
        if spot_h <= 0:
            # tell the caller this spot did not work
            return None

        # check that no other label sits inside the spot
        spot_is_clear = self._area_is_clear(spot_x1, spot_y1, spot_x2, spot_y2, bounding_boxes, chosen)
        # another label is inside the spot
        if not spot_is_clear:
            # skip this spot so we never change another label's area
            return None

        # will hold where the product area starts on each usable side
        good_starts = []

        # where the area just left of the spot would start
        left_start = spot_x1 - spot_w
        # the left area fits inside the photo
        if left_start >= 0:
            # check the left area has no label on it
            left_is_clear = self._area_is_clear(left_start, spot_y1, spot_x1, spot_y2, bounding_boxes, chosen)
            # the left area is clear, so it is product
            if left_is_clear:
                # remember the left side as a usable product area
                good_starts.append(left_start)

        # where the area just right of the spot would end
        right_end = spot_x2 + spot_w
        # the right area fits inside the photo
        if right_end <= w:
            # check the right area has no label on it
            right_is_clear = self._area_is_clear(spot_x2, spot_y1, right_end, spot_y2, bounding_boxes, chosen)
            # the right area is clear, so it is product
            if right_is_clear:
                # remember the right side as a usable product area
                good_starts.append(spot_x2)

        # neither side has clear product next to the spot
        if len(good_starts) == 0:
            # tell the caller this spot did not work
            return None

        # pick one of the usable sides at random
        product_x1 = self.rng.choice(good_starts)
        # right edge of the product area
        product_x2 = product_x1 + spot_w

        # a grey copy of the photo, used only to measure brightness
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # how bright the empty spot is on average, from 0 to 255
        spot_brightness = float(grey[spot_y1:spot_y2, spot_x1:spot_x2].mean())
        # how bright the product is on average, from 0 to 255
        product_brightness = float(grey[spot_y1:spot_y2, product_x1:product_x2].mean())

        # a spot that is fully black cannot be brightened by multiplying
        if spot_brightness < 1:
            # tell the caller this spot did not work
            return None

        # how much darker the empty spot is than the product
        difference = product_brightness - spot_brightness
        # the spot is not clearly darker, so there is no shortcut to break
        if difference < self.min_difference:
            # tell the caller this spot did not work
            return None

        # how many times brighter the spot must get to match the product
        spot_gain = product_brightness / spot_brightness
        # do not brighten more than the limit
        if spot_gain > self.max_gain:
            # use the limit instead
            spot_gain = self.max_gain

        # how many times darker the product must get to match the spot
        product_gain = spot_brightness / product_brightness
        # the smallest gain allowed, which is the most darkening allowed
        smallest_gain = 1.0 / self.max_gain
        # do not darken more than the limit
        if product_gain < smallest_gain:
            # use the limit instead
            product_gain = smallest_gain

        # work on a decimal copy so the maths does not round or overflow
        new_image = image.astype(np.float32)

        # brighten the empty spot
        self._relight(new_image, spot_x1, spot_y1, spot_x2, spot_y2, spot_gain)
        # darken the product beside it
        self._relight(new_image, product_x1, spot_y1, product_x2, spot_y2, product_gain)

        # keep every pixel inside the normal 0 to 255 range
        new_image = np.clip(new_image, 0, 255)
        # turn the decimal photo back into a normal photo
        new_image = new_image.astype(np.uint8)

        # will hold the boxes for the new photo
        new_boxes = []

        # walk through every box we were given
        for i in range(len(bounding_boxes)):
            # nothing moved, so every box is still correct, copy it over
            new_boxes.append(BoundingBox(bounding_boxes[i].type, bounding_boxes[i].box))

        # hand back the new photo and the boxes
        return new_image, new_boxes

    # makes one area brighter or darker, fading out softly at the edges
    def _relight(self, new_image, x1, y1, x2, y2, gain):
        # how tall the area is
        area_h = y2 - y1
        # how wide the area is
        area_w = x2 - x1

        # start with the edge size from the settings
        edge = self.edge
        # the fade cannot take up more than a quarter of the area's height
        if edge > area_h // 4:
            # shrink the fade to fit
            edge = area_h // 4
        # the fade cannot take up more than a quarter of the area's width
        if edge > area_w // 4:
            # shrink the fade to fit
            edge = area_w // 4

        # a map of how strongly to change each pixel, starting at zero
        weights = np.zeros((area_h, area_w), dtype=np.float32)
        # the middle of the area gets the full change
        weights[edge:area_h - edge, edge:area_w - edge] = 1.0

        # only blur when there is a fade to make
        if edge > 0:
            # blur size must be an odd number so it is centered
            blur_size = edge * 2 + 1
            # blur the map so the change fades out smoothly at the edges
            weights = cv2.GaussianBlur(weights, (blur_size, blur_size), 0)

        # how much to multiply each pixel by, 1 means no change
        factors = 1.0 + weights * (gain - 1.0)
        # copy the map 3 times so it covers all 3 colour channels
        factors = np.stack([factors, factors, factors], axis=2)

        # change the brightness of every pixel in the area
        new_image[y1:y2, x1:x2] = new_image[y1:y2, x1:x2] * factors

    # checks that no label, other than the one we skip, overlaps an area
    def _area_is_clear(self, x1, y1, x2, y2, bounding_boxes, skip):
        # walk through every box
        for i in range(len(bounding_boxes)):
            # this is the spot we are working on
            if i == skip:
                # ignore it and move to the next box
                continue

            # grab this box
            box = bounding_boxes[i]
            # left edge of this box
            box_x1 = box.box[0]
            # top edge of this box
            box_y1 = box.box[1]
            # right edge of this box
            box_x2 = box.box[0] + box.box[2]
            # bottom edge of this box
            box_y2 = box.box[1] + box.box[3]

            # this box is fully to the left of the area
            if box_x2 <= x1:
                # they do not overlap, move to the next box
                continue
            # this box is fully to the right of the area
            if box_x1 >= x2:
                # they do not overlap, move to the next box
                continue
            # this box is fully above the area
            if box_y2 <= y1:
                # they do not overlap, move to the next box
                continue
            # this box is fully below the area
            if box_y1 >= y2:
                # they do not overlap, move to the next box
                continue

            # none of the checks above passed, so this box overlaps the area
            return False

        # no box overlapped the area
        return True
