import os
import cv2
import supervision as sv
from tqdm import tqdm
from .utils import enhance_class_name, segment, load_models


def single_image_annotation(
    project_dir: str,
    source_image_path: str,
    classes: list,
    box_threshold: float,
    text_threshold: float,
    device,
) -> None:
    config_path = os.path.join(
        project_dir, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    checkpoint_path = os.path.join(project_dir, "weights/groundingdino_swint_ogc.pth")
    sam_checkpoint_path = os.path.join(project_dir, "weights/sam_vit_h_4b8939.pth")
    grounding_dino_model, sam_predictor = load_models(
        config_path, checkpoint_path, sam_checkpoint_path, device
    )

    image = cv2.imread(source_image_path)
    detections = grounding_dino_model.predict_with_classes(
        image=image,
        classes=enhance_class_name(class_names=classes),
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    # ... (code for single image mask auto-annotation)


def dataset_annotation(
    project_dir: str,
    images_directory: str,
    images_extensions: list,
    classes: list,
    box_threshold: float,
    text_threshold: float,
    device,
) -> dict:
    config_path = os.path.join(
        project_dir, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    checkpoint_path = os.path.join(project_dir, "weights/groundingdino_swint_ogc.pth")
    sam_checkpoint_path = os.path.join(project_dir, "weights/sam_vit_h_4b8939.pth")
    grounding_dino_model, sam_predictor = load_models(
        config_path, checkpoint_path, sam_checkpoint_path, device
    )

    images = {}
    annotations = {}

    image_paths = sv.list_files_with_extensions(
        directory=images_directory, extensions=images_extensions
    )

    for image_path in tqdm(image_paths):
        image_name = image_path.name
        image_path = str(image_path)
        image = cv2.imread(image_path)

        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=classes),
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )
        detections = detections[detections.class_id != None]
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )
        images[image_name] = image
        annotations[image_name] = detections

    return images, annotations
