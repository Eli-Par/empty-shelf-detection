import cv2
from cv2 import Mat

from data_classes import BoundingBox


def render(
    image: Mat,
    bounding_boxes: list[BoundingBox],
    type_names: dict[int, str],
    type_colours: dict[int, str],
) -> Mat:
    image_copy = image.copy()
    for bounding_box in bounding_boxes:
        x, y, width, height = bounding_box.box
        type_name = type_names.get(bounding_box.type, str(bounding_box.type))
        colour = type_colours.get(bounding_box.type, (255, 0, 0))

        cv2.rectangle(image_copy, (x, y), (x + width, y + height), colour, 1)

        (text_width, text_height), baseline = cv2.getTextSize(type_name, cv2.FONT_HERSHEY_PLAIN, 0.5, 1)
        cv2.rectangle(image_copy, (x, max(y - text_height - baseline, 0)), (x + text_width, y), (255, 255, 255), cv2.FILLED)
        cv2.rectangle(image_copy, (x, max(y - text_height - baseline, 0)), (x + text_width, y), colour, 1)
        cv2.putText(image_copy, type_name, (x, max(y - baseline, text_height)), cv2.FONT_HERSHEY_PLAIN, 0.5, colour, 1)

    return image_copy