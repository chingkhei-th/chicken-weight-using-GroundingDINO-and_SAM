# src/calibration.py

import cv2
import numpy as np
import supervision as sv
from src.utils import enhance_class_name, segment, calculate_area


# Constants
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297


def calculate_actual_a4_area():
    """Calculate the actual area of an A4 paper in square millimeters."""
    return A4_WIDTH_MM * A4_HEIGHT_MM


def mask_to_xyxy(mask):
    """Convert a binary mask to xyxy coordinates."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return np.array([x1, y1, x2, y2])


def detect_and_segment_paper(
    grounding_dino_model, sam_predictor, image, box_threshold, text_threshold
):
    """Detect and segment the A4 paper in the image."""
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(["paper"]),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    if len(detections.xyxy) == 0:
        raise ValueError("No paper detected in the image.")

    masks = segment(
        sam_predictor=sam_predictor,
        image=image,
        xyxy=detections.xyxy,
    )

    if len(masks) == 0:
        raise ValueError("Segmentation failed to produce any masks.")

    # Create a full image-sized mask
    full_mask = np.zeros(image.shape[:2], dtype=bool)
    full_mask[masks[0] > 0] = True

    print(f"Mask shape: {full_mask.shape}")
    print(f"Mask dtype: {full_mask.dtype}")
    print(f"Mask min-max: {full_mask.min()}-{full_mask.max()}")

    return full_mask


def calculate_calibration_factor(actual_area, detected_area):
    """Calculate the calibration factor."""
    return actual_area / detected_area


def apply_calibration(area, calibration_factor):
    """Apply the calibration factor to an area."""
    return area * calibration_factor


def annotate_paper(image, mask, area, calibrated_area):
    """Annotate the paper with a bounding box and area information."""
    print(f"Image shape: {image.shape}")
    print(f"Mask shape in annotate_paper: {mask.shape}")
    print(f"Mask dtype in annotate_paper: {mask.dtype}")
    print(f"Mask min-max in annotate_paper: {mask.min()}-{mask.max()}")

    xyxy = mask_to_xyxy(mask)
    detections = sv.Detections(
        xyxy=xyxy[None, ...],
        mask=mask[None, ...],
    )

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    label = f"Paper - Area: {area:.2f} px, Calibrated: {calibrated_area:.2f} sq mm"

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(
        scene=annotated_image, detections=detections, labels=[label]
    )

    return annotated_image
