# src/calibration_cal7.py

import cv2
import numpy as np
import supervision as sv
from src.utils import segment

# Constants
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297
A4_ASPECT_RATIO = A4_WIDTH_MM / A4_HEIGHT_MM


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
        classes=["paper"],
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

    full_mask = np.zeros(image.shape[:2], dtype=bool)
    full_mask[masks[0] > 0] = True

    return full_mask


def calculate_calibration_factor(paper_mask):
    """Calculate the pixels per mm based on the A4 paper width."""
    xyxy = mask_to_xyxy(paper_mask)
    paper_width_px = xyxy[2] - xyxy[0]
    paper_height_px = xyxy[3] - xyxy[1]

    # Check if the paper is in landscape or portrait orientation
    if paper_width_px / paper_height_px > 1:
        pixels_per_mm = paper_width_px / A4_WIDTH_MM
    else:
        pixels_per_mm = paper_width_px / A4_HEIGHT_MM

    return pixels_per_mm


def apply_calibration(area_px, pixels_per_mm):
    """Convert pixel area to square millimeters."""
    return area_px / (pixels_per_mm**2)


def annotate_paper(image, mask, area_px, area_mm):
    """Annotate the paper with a bounding box and area information."""
    xyxy = mask_to_xyxy(mask)
    detections = sv.Detections(
        xyxy=xyxy[None, ...],
        mask=mask[None, ...],
    )

    box_annotator = sv.BoxAnnotator()
    mask_annotator = sv.MaskAnnotator()

    label = f"Paper - Area: {area_px:.2f} px, Calibrated: {area_mm:.2f} sq mm"

    annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
    annotated_image = box_annotator.annotate(
        scene=annotated_image, detections=detections, labels=[label]
    )

    return annotated_image
