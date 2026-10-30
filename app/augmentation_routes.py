import base64

import cv2
from flask import Blueprint, json, jsonify, request
import numpy as np

from app.app_util import load_samples
from app.app_util import image_b64
from augmentations.move_augmention import MoveAugmentation
from augmentations.move_multi_augmention import MoveMultiAugmentation
from augmentations.oos_mirroring_augmentation import OosMirroringAugmentation
from augmentations.oos_mirroring_repeated_augmentation import OosMirroringRepeatedAugmentation
from data_classes import BoundingBox

augmentation = Blueprint("augmentation", __name__)

augmentations = {
    "OOS Mirror": lambda: OosMirroringAugmentation(seed=None),
    "OOS Mirror 2N": lambda: OosMirroringRepeatedAugmentation(seed=None),
    "Move": lambda: MoveAugmentation(seed=None),
    "Move Multi": lambda: MoveMultiAugmentation(seed=None),
}

@augmentation.route("/api/augmentation/augmentation_options", methods=["GET"])
def augmentation_options():
    return jsonify([key for key in augmentations])

@augmentation.route("/api/augmentation/image_options", methods=["GET"])
def image_options():
    images, rendered_images, boxes = load_samples("prediction", render_boxes=True) #TODO change split
    
    return jsonify([
        {
            "image": img,
            "render": render,
            "boxes": [
                {
                    "type": str(box.type),
                    "box": {
                        "x": int(box.box[0]),
                        "y": int(box.box[1]),
                        "width": int(box.box[2]),
                        "height": int(box.box[3]),
                    },
                }
                for box in sample_boxes
            ],
        }
        for img, render, sample_boxes in zip(images, rendered_images, boxes)
    ]), 200

@augmentation.route("/api/augmentation/augment", methods=["POST"])
def augment():
    augmentation = request.form.get("augmentation")
    boxes_json = request.form.get("boxes")

    if augmentation not in augmentations:
        return jsonify({
            "error": 400,
            "message": f"Unknown augmentation: {augmentation}"
        }), 400

    if boxes_json is None:
            return jsonify({
                "error": 400,
                "message": f"Unknown augmentation: {augmentation}"
            }), 400

    try:
        boxes_data = json.loads(boxes_json)

        if not isinstance(boxes_data, list):
            raise ValueError("Boxes must be a JSON array")

        bounding_boxes = []
        for item in boxes_data:
            box_type = int(item["type"])
            box = item["box"]

            x = int(box["x"])
            y = int(box["y"])
            width = int(box["width"])
            height = int(box["height"])

            bounding_boxes.append(BoundingBox(box_type, (x, y, width, height)))

    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return jsonify({
            "error": 400,
            "message": f"Invalid bounding boxes: {e}"
        }), 400

    if "image" not in request.files:
        return jsonify({
            "error": 400,
            "message": "No image provided"
        }), 400

    raw_image = request.files["image"]

    image_bytes = raw_image.read()
    image_array = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    augmentation = augmentations.get(augmentation)()

    new_image, new_bounding_boxes = augmentation.augment(image, bounding_boxes)

    return jsonify({
        "image": image_b64(new_image),
        "boxes": [
            {
                "type": str(box.type),
                "box": {
                    "x": int(box.box[0]),
                    "y": int(box.box[1]),
                    "width": int(box.box[2]),
                    "height": int(box.box[3]),
                },
            }
            for box in new_bounding_boxes
        ]
    }), 200