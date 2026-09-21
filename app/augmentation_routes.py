import base64

import cv2
from flask import Blueprint, jsonify, request
import numpy as np

from rendering.render import render

augmentation = Blueprint("prediction", __name__)

augmentations = {
    
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


@augmentation.route("/api/augmentation/augmentation_options", methods=["GET"])
def augmentation_options():
    return jsonify([key for key in augmentations])

@augmentation.route("/api/augmentation/render", methods=["POST"])
def render():
    return jsonify({
        "message": "Not Implemented"
    }), 500

@augmentation.route("/api/augmentation/augment", methods=["POST"])
def predict():
    return jsonify({
        "message": "Not Implemented"
    }), 500