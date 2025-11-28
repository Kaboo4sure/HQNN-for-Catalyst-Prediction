import os

# Detect GitHub Actions path if present
GITHUB_WORKSPACE = os.getenv("GITHUB_WORKSPACE")

if GITHUB_WORKSPACE:
    # Running inside GitHub Actions
    ROOT_DIR = os.path.abspath(GITHUB_WORKSPACE)
else:
    # Running locally or in Colab
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(ROOT_DIR, "notebooks", "data")
DRM_DIR = os.path.join(DATA_DIR, "drm")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

TARGET = "ch4_conversion"

CATALYST_CATEGORICAL = [
    "active_metal",
    "promoter",
    "support_material",
    "synthesis_method",
]
