import os
import numpy as np
from typing import List
from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [f"all {class_name}s" for class_name in class_names]


def segment(
    sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


def load_models(
    config_path: str, checkpoint_path: str, sam_checkpoint_path: str, device
) -> tuple:
    grounding_dino_model = Model(
        model_config_path=config_path, model_checkpoint_path=checkpoint_path
    )
    sam_encoder_version = "vit_h"
    sam = sam_model_registry[sam_encoder_version](checkpoint=sam_checkpoint_path).to(
        device=device
    )
    sam_predictor = SamPredictor(sam)
    return grounding_dino_model, sam_predictor


def calculate_area(mask):
    """
    Calculate the area of a segmented object in pixels.

    Args:
        mask (numpy.ndarray): The binary mask of the segmented object.

    Returns:
        int: The area of the segmented object in pixels.
    """
    area = np.sum(mask)
    return area
