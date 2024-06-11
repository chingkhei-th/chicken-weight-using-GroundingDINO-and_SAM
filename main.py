import os
from tqdm import tqdm
import roboflow
from roboflow import Roboflow

# Set project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Upload annotations to Roboflow
PROJECT_NAME = "auto-generated-dataset-7"
PROJECT_DESCRIPTION = "auto-generated-dataset-7"

roboflow.login()
workspace = Roboflow().workspace()
new_project = workspace.create_project(
    project_name=PROJECT_NAME,
    project_license="MIT",
    project_type="instance-segmentation",
    annotation=PROJECT_DESCRIPTION,
)

ANNOTATIONS_DIRECTORY = os.path.join(PROJECT_DIR, "data", "annotations")
image_paths = sv.list_files_with_extensions(
    directory=os.path.join(PROJECT_DIR, "data", "images"),
    extensions=["jpg", "jpeg", "png"],
)

for image_path in tqdm(image_paths):
    image_name = image_path.name
    annotation_name = f"{image_path.stem}.xml"
    image_path = str(image_path)
    annotation_path = os.path.join(ANNOTATIONS_DIRECTORY, annotation_name)
    new_project.upload(
        image_path=image_path,
        annotation_path=annotation_path,
        split="train",
        is_prediction=True,
        overwrite=True,
        tag_names=["auto-annotated-with-grounded-sam"],
        batch_name="auto-annotated-with-grounded-sam",
    )
