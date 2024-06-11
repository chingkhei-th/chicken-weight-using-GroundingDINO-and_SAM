import os
import cv2
import supervision as sv
from tqdm import tqdm
from .utils import segment, load_models


def convert_detection_to_segmentation(
    project_dir: str, dataset_location: str, device
) -> None:
    config_path = os.path.join(
        project_dir, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    checkpoint_path = os.path.join(project_dir, "weights/groundingdino_swint_ogc.pth")
    sam_checkpoint_path = os.path.join(project_dir, "weights/sam_vit_h_4b8939.pth")
    grounding_dino_model, sam_predictor = load_models(
        config_path, checkpoint_path, sam_checkpoint_path, device
    )

    object_detection_dataset = sv.Dataset.from_pascal_voc(
        images_directory_path=os.path.join(dataset_location, "train"),
        annotations_directory_path=os.path.join(dataset_location, "train"),
    )

    for image_name, image in tqdm(object_detection_dataset.images.items()):
        detections = object_detection_dataset.annotations[image_name]
