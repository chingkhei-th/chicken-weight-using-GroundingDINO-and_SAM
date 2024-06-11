import os
import sys


def download_models(project_dir):
    """
    Download Grounding DINO and Segment Anything Model (SAM) weights.

    Args:
        project_dir (str): Path to the project directory.
    """
    # Create the weights directory if it doesn't exist
    weights_dir = os.path.join(project_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    # Change the current working directory to the project directory
    os.chdir(project_dir)

    # Install Grounding DINO
    os.system("git clone https://github.com/IDEA-Research/GroundingDINO.git")
    os.chdir("GroundingDINO")
    os.system("git checkout -q 57535c5a79791cb76e36fdb64975271354f10251")
    os.system("pip install -q -e .")
    os.chdir(project_dir)

    # Install Segment Anything Model
    os.system(
        f"{sys.executable} -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'"
    )

    # Download Grounding DINO model weights
    os.chdir(weights_dir)
    os.system(
        "wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    )

    # Download Segment Anything Model (SAM) weights
    os.system(
        "wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    )


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.abspath(__file__))
    download_models(project_dir)
