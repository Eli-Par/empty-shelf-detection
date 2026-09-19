from prediction.yolov8_predict import YoloV8Predict
from rendering.render import render
import cv2
import tkinter as tk
from tkinter import filedialog

type_names = {
    0: "full",
    1: "partial",
}

type_colours = {
    0: (0, 255, 0),
    1: (0, 0, 255),
}

def main():
    print("Starting tinker")
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()

    print("File picker closed")
    if file_path is None:
        print("No file found")
        return

    print("Instantiating prediction object")
    predictor = YoloV8Predict("Standard plus OOS Mirroring")

    print("Loading image")
    image = cv2.imread(file_path)

    print("Image loaded")
    if image is None:
            print("No image found")
            return

    print("Predicting...")
    boxes = predictor.predict(image)

    # for box in boxes:
    #     print(box.box, box.type)

    print("Rendering prediction")
    image = render(image, boxes, type_names, type_colours)

    print("Showing prediction")
    cv2.imshow("Prediction", image)

    cv2.waitKey(0)

main()