import os
import torch
import joblib

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from sklearn.linear_model import LinearRegression
from src.utils import load_models
from src.annotate import annotate_and_save_images
from src.plot import plot_and_save_images
from src.weight_predict import predict_and_display_weights

# Define directories
TEST_IMAGES_DIR = "./data/images/test"
OUTPUT_DIR = "./data/outputs"

# Define other parameters
CLASSES = ["chicken"]
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

# Load the linear regression model
linear_regression_model = joblib.load("./regression-model/LRmodel.pkl")

# Annotate and save images
annotate_and_save_images(
    grounding_dino_model,
    sam_predictor,
    TEST_IMAGES_DIR,
    os.path.join(OUTPUT_DIR, "annotated"),
    CLASSES,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
)

# Plot and save images
plot_and_save_images(
    grounding_dino_model,
    sam_predictor,
    TEST_IMAGES_DIR,
    os.path.join(OUTPUT_DIR, "plots"),
    CLASSES,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
)

# Predict and display weights
predict_and_display_weights(
    grounding_dino_model,
    sam_predictor,
    TEST_IMAGES_DIR,
    os.path.join(OUTPUT_DIR, "weights"),
    CLASSES,
    BOX_THRESHOLD,
    TEXT_THRESHOLD,
    linear_regression_model,
)
