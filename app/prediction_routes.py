import base64
import os

import cv2
from flask import Blueprint, jsonify, request
import numpy as np

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

    predictor = models[model](augmentation)
    boxes = predictor.predict(image)

    rendered_image = render(image, boxes, type_names, type_colours)

    success, buffer = cv2.imencode(".png", rendered_image)

    if not success:
        return jsonify({
            "error": 500,
            "message": "Failed to encode rendered image"
        }), 500

    image_base64 = base64.b64encode(buffer).decode("utf-8")
    
    return jsonify({
        "image": image_base64,
        "boxes": [
            {
                "type": box.type,
                "box": {
                    "x": box.box[0],
                    "y": box.box[1],
                    "width": box.box[2],
                    "height": box.box[3],
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