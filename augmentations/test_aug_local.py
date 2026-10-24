
import tkinter as tk
from tkinter import filedialog

import cv2

from augmentations.move_augmention import MoveAugmentation
from augmentations.move_multi_augmention import MoveMultiAugmentation
from data_loader.loader import Loader
from rendering.render import render


type_names = {
    0: "full",
    1: "partial",
}

type_colours = {
    0: (0, 255, 0),
    1: (0, 0, 255),
}


def main():
    print("Starting augmentation test")

    root = tk.Tk()
    root.withdraw()

    yaml_path = filedialog.askopenfilename(
        title="Select data.yaml",
        filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
    )

    if not yaml_path:
        print("No YAML file selected")
        return

    split = "train"

    print(f"Loading {split} split from: {yaml_path}")
    loader = Loader(yaml_path)
    images, all_boxes = loader.load(split)

    if not images:
        print("No images found in the split")
        return

    image = images[0]
    boxes = all_boxes[0]

    print(f"Loaded first image ({image.shape[1]}x{image.shape[0]})")
    print(f"Found {len(boxes)} bounding boxes")

    if not boxes:
        print("The first image has no bounding boxes to augment")
        return

    while True:
        print("Applying MoveMultiAugmentation...")
        augmentation = MoveMultiAugmentation(
            # full_to_partial_rate=1,
            # max_move_dist=0.75,
            # max_tries=1000,
            seed=None,
        )

        result = augmentation.augment(image, boxes)

        augmented_image, augmented_boxes = result

        original_rendered = render(image.copy(), boxes, type_names, type_colours)

        augmented_rendered = render(augmented_image.copy(), augmented_boxes, type_names, type_colours)

        print("Showing original and augmented images")
        cv2.imshow("Original Image", original_rendered)
        cv2.imshow("Augmented Image", augmented_rendered)
        
        print("Press any key in an image window to exit")
        if cv2.waitKey(0) == 27:
            break


if __name__ == "__main__":
    main()
