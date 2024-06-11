import os
import cv2
import numpy as np


from utils import enhance_class_name, segment, calculate_area


def predict_and_display_weights(
    grounding_dino_model,
    sam_predictor,
    image_dir,
    output_dir,
    classes,
    box_threshold,
    text_threshold,
    linear_regression_model,
):
    """
    Predict and display the weights of detected objects.
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

            areas = [calculate_area(mask) for mask in detections.mask]

            for class_id, area in zip(detections.class_id, areas):
                weight = linear_regression_model.predict([[area]])[0]
                print(
                    f"{classes[class_id]}: Area = {area} pixels, Predicted weight = {weight:.2f} g"
                )
            print()
