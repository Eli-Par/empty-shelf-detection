from training.yolo_train import YoloTrain

class Yolov11Train(YoloTrain):

    def __init__(self, train_images, train_boxes, validate_images, validate_boxes, test_images, test_boxes, epochs=50, batch_size=16, img_size=640, lr0=0.01):
        super().__init__("yolo11n.pt", train_images, train_boxes, validate_images, validate_boxes, test_images, test_boxes, epochs, batch_size, img_size, lr0)