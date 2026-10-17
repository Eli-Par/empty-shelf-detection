import random

import cv2

from augmentations.interface.augmentation import Augmentation
from data_classes import BoundingBox


class MoveAugmentation(Augmentation):

    def __init__(self, full_to_partial_rate=0.75, max_move_dist=0.75, min_height=0.25, max_height=0.6, max_tries=10000, seed=100):
        self.full_to_partial_rate = full_to_partial_rate
        self.max_move_dist = max_move_dist
        self.min_height = min_height
        self.max_height = max_height
        self.max_tries = max_tries
        self.rng = random.Random(seed)

    def augment(self, image: cv2.Mat, bounding_boxes: list[BoundingBox]):
        h = image.shape[0]
        w = image.shape[1]

        (content_left, content_right) = self._get_content_bounds(image, 255)
        content_width = content_right - content_left

        bounding_box = None

        # repeat until a valid box is found or timeout
        count = 0
        while count < self.max_tries:
            count += 1

            # pick a random box
            bounding_box = self.rng.choice(bounding_boxes)

            original_box_type = bounding_box.type

            bounding_box_bottom = bounding_box.box[1] + bounding_box.box[3]

            # determine the new bounding box type. Partial stays partial. Full can become partial at the specified rate if in the bottom half of the image
            new_type = (
                1 if original_box_type == 1
                else (
                    int(self.rng.random() < self.full_to_partial_rate)
                    if bounding_box_bottom >= h / 2
                    else original_box_type
                )
            )

            new_height = bounding_box.box[3]
            if original_box_type == 0 and new_type == 1:
                new_height = self.rng.randint(int(bounding_box.box[3] * self.min_height), int(bounding_box.box[3] * self.max_height))

            # calculate min and max x position
            move_min = int(max(
                bounding_box.box[0] - self.max_move_dist * content_width, 
                content_left, 
                0
            ))
            move_max = int(min(
                bounding_box.box[0] + (self.max_move_dist * content_width) - bounding_box.box[2], 
                content_right - bounding_box.box[2], 
                w - bounding_box.box[2]
            ))

            new_x = self.rng.randint(move_min, move_max)

            new_y = bounding_box.box[1] + (bounding_box.box[3] - new_height)

            new_bounding_box = BoundingBox(new_type, (new_x, new_y, bounding_box.box[2], new_height))

            if not self._overlaps_any(new_bounding_box, bounding_boxes, new_type, 0.2):
                # once successful bounding box found, copy pixels and adjust bounding box list

                new_image = image.copy()
                new_boxes = []

                # fill original bounding box with pixels from new bounding box position
                self._fill_original(image, new_image, bounding_box, new_x)

                # copy pixels from original bounding box into new bounding box, which is aligned at the bottom
                self._fill_new(image, new_image, bounding_box, new_x, new_y, new_height)

                # add all bounding boxes, except the removed box, append the new box and shuffle
                new_boxes = [box for box in bounding_boxes if box is not bounding_box]
                new_boxes.append(new_bounding_box)
                self.rng.shuffle(new_boxes)

                return new_image, new_boxes

    def _fill_original(self, image: cv2.Mat, new_image: cv2.Mat, bounding_box: BoundingBox, new_x: int):
        x, y, width, height = bounding_box.box
        new_image[y:y + height, x:x + width] = image[y:y + height, new_x:new_x + width].copy()

    def _fill_new(self, image: cv2.Mat, new_image: cv2.Mat, bounding_box: BoundingBox, new_x: int, new_y: int, new_height: int):
        x, y, width, height = bounding_box.box
        src_y = y + height - new_height

        new_image[new_y:new_y + new_height, new_x:new_x + width] = image[src_y:src_y + new_height, x:x + width].copy()

    def _overlaps_any(self, new_box: BoundingBox, existing_boxes: list[BoundingBox], type: int, x_threshold: float) -> bool:
        x1, y1, w1, h1 = new_box.box

        for box in existing_boxes:

            x2, y2, w2, h2 = box.box

            if type == box.type:
                padding = int(w1 * x_threshold)
                x1_adjusted = x1 - padding
                w1_adjusted = w1 + 2 * padding
            else:
                x1_adjusted = x1
                w1_adjusted = w1

            if (x1_adjusted < x2 + w2 and x1_adjusted + w1_adjusted > x2 and y1 < y2 + h2 and y1 + h1 > y2):
                return True

        return False

    def _get_content_bounds(self, image: cv2.Mat, threshold: int = 255):
        # Determine which pixels are below the threshold
        if len(image.shape) == 2:
            content_mask = image < threshold
        else:
            content_mask = (image[:, :, :3] < threshold).all(axis=2)

        # Find columns that have a pixel below the thershold
        content_columns = content_mask.any(axis=0)
        left = 0
        right = image.shape[1]

        for x in range(len(content_columns)):
            if content_columns[x]:
                left = x
                break

        for x in range(len(content_columns) - 1, -1, -1):
            if content_columns[x]:
                right = x + 1
                break

        return left, right