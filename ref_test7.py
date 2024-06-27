# main.py

import os
import cv2
import numpy as np
import torch
import supervision as sv
from src.utils import (load_models, enhance_class_name, segment)
from src.annotate import annotate_and_save_images
from src.plot import plot_and_save_images
from src.calibration_cal7 import (
    detect_and_segment_paper,
    calculate_calibration_factor,
    apply_calibration,
    annotate_paper,
)

# Define directories
REFERENCE_IMAGE_DIR = "./data/test/"
phone_IMAGE_DIR = "./data/test/phone/"
OUTPUT_DIR = "./data/outputs_7.2/"

# Define other parameters
CLASSES = ["paper", "phone"]
BOX_THRESHOLD = 0.35
TEXT_THRESHOLD = 0.25

# Load models
config_path = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
checkpoint_path = "./weights/groundingdino_swint_ogc.pth"
sam_checkpoint_path = "./weights/sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"

grounding_dino_model, sam_predictor = load_models(
    config_path, checkpoint_path, sam_checkpoint_path, device
)


# Calibration process
def calibrate():
    reference_image_path = os.path.join(REFERENCE_IMAGE_DIR, "test_paper.jpg")
    reference_image = cv2.imread(reference_image_path)
    reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

    paper_mask = detect_and_segment_paper(
        grounding_dino_model,
        sam_predictor,
        reference_image_rgb,
        BOX_THRESHOLD,
        TEXT_THRESHOLD,
    )

    pixels_per_mm = calculate_calibration_factor(paper_mask)

    detected_paper_area_px = np.sum(paper_mask)
    calibrated_paper_area_mm = apply_calibration(detected_paper_area_px, pixels_per_mm)

    print(f"Detected paper area: {detected_paper_area_px} pixels")
    print(f"Calibrated paper area: {calibrated_paper_area_mm:.2f} sq mm")
    print(f"Pixels per mm: {pixels_per_mm:.2f}")

    annotated_reference = annotate_paper(
        reference_image, paper_mask, detected_paper_area_px, calibrated_paper_area_mm
    )
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "annotated_reference.jpg"), annotated_reference
    )

    return pixels_per_mm


def main():
    pixels_per_mm = calibrate()

    # Process phone images
    for filename in os.listdir(phone_IMAGE_DIR):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(phone_IMAGE_DIR, filename)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            detections = grounding_dino_model.predict_with_classes(
                image=image_rgb,
                classes=["phone"],
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=image_rgb,
                xyxy=detections.xyxy,
            )

            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()

            labels = []
            for i, mask in enumerate(detections.mask):
                phone_area_px = np.sum(mask)
                phone_area_mm = apply_calibration(phone_area_px, pixels_per_mm)
                label = f"Phone {i+1} - Area: {phone_area_px:.2f} px, Calibrated: {phone_area_mm:.2f} sq mm"
                labels.append(label)

                print(f"Phone {i+1} in {filename}:")
                print(f"  Detected area: {phone_area_px} pixels")
                print(f"  Calibrated area: {phone_area_mm:.2f} sq mm")

            annotated_image = mask_annotator.annotate(
                scene=image.copy(), detections=detections
            )
            annotated_image = box_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels
            )

            output_path = os.path.join(
                OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_annotated.jpg"
            )
            cv2.imwrite(output_path, annotated_image)

            segmented_output = np.max(detections.mask, axis=0) * 255
            segmented_output_path = os.path.join(
                OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_segmented.png"
            )
            cv2.imwrite(segmented_output_path, segmented_output)

    # Annotate and save images
    annotate_and_save_images(
        grounding_dino_model,
        sam_predictor,
        phone_IMAGE_DIR,
        os.path.join(OUTPUT_DIR, "annotated"),
        ["phone"],
        BOX_THRESHOLD,
        TEXT_THRESHOLD,
    )

    # Plot and save images
    plot_and_save_images(
        grounding_dino_model,
        sam_predictor,
        phone_IMAGE_DIR,
        os.path.join(OUTPUT_DIR, "plots"),
        ["phone"],
        BOX_THRESHOLD,
        TEXT_THRESHOLD,
    )


if __name__ == "__main__":
    main()
