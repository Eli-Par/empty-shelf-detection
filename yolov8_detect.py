import cv2
import sys
import json
from ultralytics import YOLO


def run_inference(image_path, model_path="models/yolov8_best.pt"):
    # Load model
    model = YOLO(model_path)

    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Run inference
    results = model(image)[0]

    detections = []

    # Parse results
    for box in results.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls = int(box.cls[0])

        detections.append({
            "class_id": cls,
            "class_name": model.names[cls],
            "confidence": conf,
            "box_xyxy": [int(x1), int(y1), int(x2), int(y2)]
        })

    # -------------------------
    # Print ALL detections
    # -------------------------
    print("\n=== DETECTIONS ===")
    print(json.dumps(detections, indent=2))

    # -------------------------
    # Draw ALL boxes
    # -------------------------
    for d in detections:
        x1, y1, x2, y2 = d["box_xyxy"]
        label = f'{d["class_name"]} {d["confidence"]:.2f}'

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show result
    cv2.imshow("YOLOv8 Detection", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# -------------------------
# CLI ENTRY POINT
# -------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python yolov8_detect.py <image_path> [model_path]")
        sys.exit(1)

    image_path = sys.argv[1]

    model_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "models/yolov8_best.pt"
    )

    run_inference(image_path, model_path)