import os
import cv2
import numpy as np
import supervision as sv

from src.utils import enhance_class_name, segment


def annotate_and_save_images(
    grounding_dino_model,
    sam_predictor,
    image_dir,
    output_dir,
    classes,
    box_threshold,
    text_threshold,
):
    """
    Annotate images with detections and save the annotated images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Loop over all images in the test directory
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Construct full path to the current image
            image_path = os.path.join(image_dir, filename)
            # Load the image
            image = cv2.imread(image_path)

            # Detect objects
            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes=enhance_class_name(classes),
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )

            # Convert detections to masks
            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                xyxy=detections.xyxy,
            )

            # Annotate image with detections
            mask_annotator = sv.MaskAnnotator()

            labels = [
                f"{classes[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _ in detections
            ]

            annotated_image = mask_annotator.annotate(
                scene=image.copy(), detections=detections
            )

            output_image_path = os.path.join(
                output_dir, f"{os.path.splitext(filename)[0]}_annotated.jpg"
            )
            cv2.imwrite(output_image_path, annotated_image)
            print(f"Processed image: {filename}")
