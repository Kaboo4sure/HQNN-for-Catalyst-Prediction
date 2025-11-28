import os

# Path to src/ folder
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Go two levels up: src → project → repo root
ROOT_DIR = os.path.abspath(os.path.join(CONFIG_DIR, "..", ".."))

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "notebooks", "data")
DRM_DIR = os.path.join(DATA_DIR, "drm")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Ensure processed dir exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

TARGET = "ch4_conversion"

CATALYST_CATEGORICAL = [
    "active_metal",
    "promoter",
    "support_material",
    "synthesis_method",
]
