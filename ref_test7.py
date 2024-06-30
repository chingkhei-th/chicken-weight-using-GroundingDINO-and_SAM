# main.py

import os
import cv2
import numpy as np
import torch
import json
import supervision as sv
from src.utils import load_models, enhance_class_name, segment
from src.annotate import annotate_and_save_images
from src.plot import plot_and_save_images
from src.calibration_cal7 import (
    calculate_actual_a4_area,
    detect_and_segment_paper,
    calculate_calibration_factor,
    apply_calibration,
    annotate_paper,
)

# Define directories
REFERENCE_IMAGE_DIR = "./data/ref/"
phone_IMAGE_DIR = "./data/ref/phone/"
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
    reference_image_path = os.path.join(REFERENCE_IMAGE_DIR, "ref_paper.jpg")
    reference_image = cv2.imread(reference_image_path)
    reference_image_rgb = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

    actual_a4_area = calculate_actual_a4_area()
    paper_mask = detect_and_segment_paper(
        grounding_dino_model,
        sam_predictor,
        reference_image_rgb,
        BOX_THRESHOLD,
        TEXT_THRESHOLD,
    )
    detected_paper_area = np.sum(paper_mask)

    calibration_factor = calculate_calibration_factor(
        actual_a4_area, detected_paper_area
    )

    print(f"Actual A4 paper area: {actual_a4_area} sq mm")
    print(f"Detected paper area: {detected_paper_area} pixels")
    print(f"Calibration factor: {calibration_factor}")

    # Annotate and save the reference image with paper
    annotated_reference = annotate_paper(
        reference_image, paper_mask, detected_paper_area, actual_a4_area
    )
    cv2.imwrite(
        os.path.join(OUTPUT_DIR, "annotated_reference.jpg"), annotated_reference
    )

    return calibration_factor


def save_phone_areas_to_json(calibrated_areas, output_file):
    """Saves the calibrated areas of phones to a JSON file."""
    with open(output_file, "w") as f:
        json.dump(calibrated_areas, f)


def load_existing_phone_areas(input_file):
    """Loads existing calibrated areas from a JSON file."""
    try:
        with open(input_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def swap_keys_in_dicts(old_dict, new_key_mapping):
    """
    Swaps keys in old_dict based on new_key_mapping.

    Parameters:
    - old_dict: Original dictionary with keys to be swapped.
    - new_key_mapping: Mapping of old keys to new keys.

    Returns:
    - Updated dictionary with keys swapped.
    """
    new_dict = {}
    for old_key, new_key in new_key_mapping.items():
        new_dict[new_key] = old_dict.get(old_key, {})
    return new_dict


def calculate_error_rate(existing_data, new_data):
    """Calculates the error rate for 'phone 1' as a percentage."""
    # Extract the value for 'phone 1' from the existing data
    existing_phone1_area = existing_data.get("phone 1")
    # Extract the value for 'phone 1' from the new data
    new_phone1_area = new_data.get("phone 1")

    # Check if both values exist; otherwise, handle appropriately (e.g., return 0% or raise an exception)
    if existing_phone1_area is None or new_phone1_area is None:
        print("One of the phone 1 areas does not exist.")
        return 0.0  # Or handle this case differently based on your needs

    # Calculate the difference
    difference = abs(new_phone1_area - existing_phone1_area)

    # Convert the difference to a percentage
    error_rate_percent = (
        (difference / existing_phone1_area) * 100
    ).real  # Use.real to avoid complex numbers in case of division by zero

    return error_rate_percent


def main():

    # Load existing calibrated areas
    existing_calibrated_areas = load_existing_phone_areas(
        os.path.join(OUTPUT_DIR, "calibrated_phone_area.json")
    )

    # Define mapping for swapping keys
    key_swap_mapping = {"phone 1": "phone 2", "phone 2": "phone 1"}

    # Swap keys in both dictionaries
    existing_calibrated_areas = swap_keys_in_dicts(
        existing_calibrated_areas, key_swap_mapping
    )

    calibration_factor = calibrate()

    # Initialize a global dictionary to hold calibrated areas
    global_calibrated_areas = {}

    # Process phone images
    for filename in os.listdir(phone_IMAGE_DIR):
        if filename.endswith((".jpg", ".png")):
            image_path = os.path.join(phone_IMAGE_DIR, filename)
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            detections = grounding_dino_model.predict_with_classes(
                image=image_rgb,
                classes=enhance_class_name(["phone"]),
                box_threshold=BOX_THRESHOLD,
                text_threshold=TEXT_THRESHOLD,
            )

            detections.mask = segment(
                sam_predictor=sam_predictor,
                image=image_rgb,
                xyxy=detections.xyxy,
            )

            # Annotate phones with area information
            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()

            labels = []

            for i, mask in enumerate(detections.mask):
                phone_area = np.sum(mask)
                calibrated_area = apply_calibration(phone_area, calibration_factor)
                global_calibrated_areas[f"phone {i+1}"] = calibrated_area
                label = f"phone {i+1} - Area: {phone_area:.2f} px, Calibrated: {calibrated_area:.2f} sq mm"
                labels.append(label)

                print(f"phone {i+1} in {filename}:")
                print(f"  Detected area: {phone_area} pixels")
                print(f"  Calibrated area: {calibrated_area:.2f} sq mm")

            annotated_image = mask_annotator.annotate(
                scene=image.copy(), detections=detections
            )
            annotated_image = box_annotator.annotate(
                scene=annotated_image, detections=detections, labels=labels
            )

            # Save the annotated output
            output_path = os.path.join(
                OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_annotated.jpg"
            )
            cv2.imwrite(output_path, annotated_image)

            # Save the segmented output
            segmented_output = np.max(detections.mask, axis=0) * 255
            segmented_output_path = os.path.join(
                OUTPUT_DIR, f"{os.path.splitext(filename)[0]}_segmented.png"
            )
            cv2.imwrite(segmented_output_path, segmented_output)

            # Calculate error rate considering the swapped keys
            error_rate_percent = calculate_error_rate(
                existing_calibrated_areas, global_calibrated_areas
            )
            print(f"Error rate: {error_rate_percent:.2f}%")

            # Save the global calibrated areas to a single JSON file
            output_json_file = os.path.join(OUTPUT_DIR, "calibrated_phone_area.json")
            save_phone_areas_to_json(global_calibrated_areas, output_json_file)

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
