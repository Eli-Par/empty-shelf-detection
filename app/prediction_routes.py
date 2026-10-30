import base64
import os

import cv2
from flask import Blueprint, jsonify, request
import numpy as np

from app.app_util import add_letterboxing, load_samples, remove_letterboxing
from data_classes import BoundingBox
from data_loader.loader import Loader
from prediction.yolov8_predict import YoloV8Predict
from prediction.yolov11_predict import YoloV11Predict
from prediction.yolov26_predict import YoloV26Predict
from prediction.rtdetr_predict import RtDetrPredict
from rendering.render import render

prediction = Blueprint("prediction", __name__)

models = {
    "yolov8n": lambda augmentation: YoloV8Predict(augmentation, 0.25, 0.1),
    "yolo11n": lambda augmentation: YoloV11Predict(augmentation, 0.25, 0.1),
    "yolo26n": lambda augmentation: YoloV26Predict(augmentation, 0.25, 0.1),
    "rt_detr": lambda augmentation: RtDetrPredict(augmentation, 0.25, 0.1),
}

MODEL_PATH = "trained_models"

type_names = {
    0: "Full",
    1: "Partial",
}

type_colours = {
    0: (40, 200, 00),
    1: (0, 0, 255),
}


@prediction.route("/api/prediction/model_options", methods=["GET"])
def model_options():
    return jsonify([key for key in models.keys()])

@prediction.route("/api/prediction/augmentation_options", methods=["GET"])
def augmentation_options():
    return jsonify(get_model_folders())

@prediction.route("/api/prediction/image_options", methods=["GET"])
def image_options():
    images, _, _ = load_samples("prediction")

    return jsonify(images)


@prediction.route("/api/prediction/predict", methods=["POST"])
def predict():
    model = request.form.get("model")
    augmentation = request.form.get("augmentation")

    if model not in models:
        return jsonify({
            "error": 400,
            "message": f"Unknown model: {model}"
        }), 400
    
    if augmentation not in get_model_folders():
        return jsonify({
            "error": 400,
            "message": f"Unknown augmentation: {augmentation}"
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

    if image is None:
        return jsonify({
            "error": 400,
            "message": "Invalid image"
        }), 400

    letterboxed_image = add_letterboxing(image)

    predictor = models[model](augmentation)
    boxes = predictor.predict(letterboxed_image)

    # Calculate scale and centered padding
    original_height, original_width = image.shape[:2]
    letterboxed_height, letterboxed_width = letterboxed_image.shape[:2]
    scale = min(letterboxed_width / original_width, letterboxed_height / original_height)

    pad_x = (letterboxed_width - original_width * scale) / 2
    pad_y = (letterboxed_height - original_height * scale) / 2

    new_boxes = []
    for box in boxes:
        x, y, width, height = box.box

        x1 = round((x - pad_x) / scale)
        y1 = round((y - pad_y) / scale)
        x2 = round((x + width - pad_x) / scale)
        y2 = round((y + height - pad_y) / scale)

        x1 = max(0, min(original_width, x1))
        y1 = max(0, min(original_height, y1))
        x2 = max(0, min(original_width, x2))
        y2 = max(0, min(original_height, y2))

        new_boxes.append(BoundingBox(box.type, (x1, y1, max(0, x2 - x1), max(0, y2 - y1))))

    boxes = new_boxes
    
    return jsonify({
        "image": '', # image_base64
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
            for box in boxes
        ],
    })

def get_model_folders():
    folder_names = []

    for name in os.listdir(MODEL_PATH):
        if os.path.isdir(os.path.join(MODEL_PATH, name)):
            folder_names.append(name)

    return folder_names