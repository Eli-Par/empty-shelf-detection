from cv2.typing import Rect


class BoundingBox:
    def __init__(self, obj_type: int, box: Rect): # rect in pixel space, x and y in top left
        self.type = obj_type
        self.box = box