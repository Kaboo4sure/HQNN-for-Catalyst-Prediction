import os

# Path to this file (src/config.py)
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to project root: go from src/ → project root
ROOT_DIR = os.path.abspath(os.path.join(CONFIG_DIR, ".."))

# Paths inside project
DATA_DIR = os.path.join(ROOT_DIR, "notebooks", "data")
DRM_DIR = os.path.join(DATA_DIR, "drm")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Ensure processed dir exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Target column
TARGET = "ch4_conversion"

# Categorical features
CATALYST_CATEGORICAL = [
    "active_metal",
    "promoter",
    "support_material",
    "synthesis_method",
]
