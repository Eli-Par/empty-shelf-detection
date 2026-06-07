# used to read image files from disk
import cv2
# used to read the data.yaml config file
import yaml
# used to build file paths that work on any os
from pathlib import Path

# our bounding box data class
from data_classes import BoundingBox


class Loader:
    """
    Loads the images and labels for a dataset split.
    Gives back two lists, the images and their boxes,
    lined up so images[i] goes with boxes[i].

    How to use:
        Give it the path to data.yaml, then load a split.
        Split is "train", "val", or "test".

        loader = Loader("data/162/data.yaml")
        images, boxes = loader.load("train")
    """

    def __init__(self, yaml_path: str):
        # store the path to data.yaml as a Path object
        self.yaml_path = Path(yaml_path)
        # open the yaml file
        with open(self.yaml_path) as f:
            # read the yaml into a dict
            self.config = yaml.safe_load(f)

    def load(self, split: str):
        # folder that holds data.yaml
        base = self.yaml_path.parent
        # yaml stores paths like ../train/images but our splits sit next to the yaml
        # so take just the last two parts and join them to base
        split_parts = Path(self.config[split]).parts[-2:]
        # full path to the images folder
        images_dir = (base / Path(*split_parts)).resolve()
        # labels folder sits next to images
        labels_dir = images_dir.parent / "labels"

        # will hold the loaded images
        images = []
        # will hold one box list per image
        all_boxes = []

        # go through image files in order
        for img_path in sorted(images_dir.iterdir()):
            # skip anything that is not an image
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue

            # load the image as a BGR array
            image = cv2.imread(str(img_path))
            # get height and width for the coord math
            h, w = image.shape[:2]

            # path to the matching label file
            label_path = labels_dir / (img_path.stem + ".txt")
            # will hold the boxes for this image
            boxes = []

            # some images have no label file
            if label_path.exists():
                # open the label file
                with open(label_path) as f:
                    # each line is one box
                    for line in f:
                        # split the line into values
                        parts = line.strip().split()
                        # skip blank lines
                        if not parts:
                            continue
                        # first value is the class id
                        class_id = int(parts[0])
                        # normalized center x y and width height
                        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        # convert center x to top left pixel x
                        x = int((cx - bw / 2) * w)
                        # convert center y to top left pixel y
                        y = int((cy - bh / 2) * h)
                        # convert width to pixels
                        bw = int(bw * w)
                        # convert height to pixels
                        bh = int(bh * h)
                        # store the box
                        boxes.append(BoundingBox(class_id, (x, y, bw, bh)))

            # add the image to the output
            images.append(image)
            # add this image's boxes to the output
            all_boxes.append(boxes)

        # return the two parallel lists
        return images, all_boxes
