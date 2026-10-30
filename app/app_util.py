

import base64
from typing import List

import cv2
import numpy as np

from data_classes import BoundingBox
from data_loader.loader import Loader
from rendering.render import render

type_names = {
    0: "Full",
    1: "Partial",
}

type_colours = {
    0: (0, 255, 0),
    1: (91, 244, 255),
}


def load_samples(split: str, render_boxes: bool = False):
    loader = Loader("app/data/data.yaml")
    images, boxes = loader.load(split)

    final_images = []
    final_rendered_images = []
    final_boxes = []
    for i in range(min(8, len(images))):
        image, box = remove_letterboxing(images[i], boxes[i])

        image_base64 = image_b64(image)
        
        final_images.append(image_base64)
        final_boxes.append(box)

        if render_boxes:
            scaled_image, scaled_box = scale_image(image, box)
            rendered_image = render(scaled_image, scaled_box, type_names, type_colours, show_labels=False, thickness=2)
            final_rendered_images.append(image_b64(rendered_image))

    return final_images, final_rendered_images, final_boxes
    
def remove_letterboxing(image: cv2.Mat, boxes: List[BoundingBox], threshold=235):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # find pixels below threshold
    mask = gray < threshold

    # rows and cols below theshold
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]

    if len(rows) == 0 or len(cols) == 0:
        return image, boxes

    top = rows[0]
    bottom = rows[-1] + 1
    left = cols[0]
    right = cols[-1] + 1

    new_image = image[top:bottom, left:right]

    image_height, image_width = new_image.shape[:2]

    new_boxes = []

    for box in boxes:
        x, y, width, height = box.box

        x1 = max(0, x - left)
        y1 = max(0, y - top)
        x2 = min(image_width, x + width - left)
        y2 = min(image_height, y + height - top)

        new_width = max(0, x2 - x1)
        new_height = max(0, y2 - y1)

        new_box = BoundingBox(box.type, (x1, y1, new_width, new_height))

        new_boxes.append(new_box)


    return new_image, new_boxes


def add_letterboxing(image: cv2.Mat):
    height, width = image.shape[:2]

    size = max(height, width)

    top = (size - height) // 2
    left = (size - width) // 2

    if len(image.shape) == 3:
        new_image = np.full((size, size, image.shape[2]), 255, dtype=image.dtype)
    else:
        new_image = np.full((size, size), 255, dtype=image.dtype)

    new_image[top:top + height, left:left + width] = image

    return new_image

def image_b64(image: cv2.Mat):
    success, buffer = cv2.imencode(".png", image)
    
    if not success:
        return None
    
    return base64.b64encode(buffer).decode("utf-8")\

def scale_image(image: cv2.Mat, boxes: list[BoundingBox], height: int = 128):
    original_height, original_width = image.shape[:2]
    if original_height < height:
        return image, boxes

    scale = height / original_height
    new_width = int(round(original_width * scale))

    scaled_image = cv2.resize(image, (new_width, height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )

    scaled_boxes = []
    for box in boxes:
        scaled_boxes.append(BoundingBox(box.type, (int(box.box[0] * scale), int(box.box[1] * scale), int(box.box[2] * scale), int(box.box[3] * scale))))

    return scaled_image, scaled_boxes