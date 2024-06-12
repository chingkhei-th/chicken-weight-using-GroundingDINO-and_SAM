import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.utils import enhance_class_name, segment


def plot_and_save_images(
    grounding_dino_model,
    sam_predictor,
    image_dir,
    output_dir,
    classes,
    box_threshold,
    text_threshold,
):
    """
    Plot and save the input images and segmented objects.
    """
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)

            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=enhance_class_name(classes),
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy,
            )

            fig = plt.figure(figsize=(16, 16))
            num_subplots = len(detections.mask) + 1
            grid_rows = np.ceil(np.sqrt(num_subplots)).astype(int)
            grid_cols = np.ceil(num_subplots / grid_rows).astype(int)
            gs = GridSpec(grid_rows, grid_cols, figure=fig)

            ax = fig.add_subplot(gs[0, 0])
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            ax.set_title("Input Image")
            ax.axis("off")

            titles = [classes[class_id] for class_id in detections.class_id]
            for i, (mask, title) in enumerate(zip(detections.mask, titles), start=1):
                row = (i - 1) // grid_cols
                col = (i - 1) % grid_cols
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(mask)
                ax.set_title(title)
                ax.axis("off")

            plt.subplots_adjust(wspace=0.1, hspace=0.1)

            output_file = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}_plot.png"
            )
            plt.savefig(output_file, bbox_inches="tight")
            plt.close(fig)
            print(f"Processed image: {filename}")
