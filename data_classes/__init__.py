from cv2.typing import Rect


class BoundingBox:
    def __init__(self, obj_type: int, box: Rect):
        self.type = obj_type
        self.box = box