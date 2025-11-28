import os

# Path to the project root (one level above /src)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# notebooks/data root
DATA_DIR = os.path.join(BASE_DIR, "notebooks", "data")

# Correct subdirectories
DRM_DIR = os.path.join(DATA_DIR, "drm")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Ensure processed folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Target variable for models
TARGET = "ch4_conversion"

CATALYST_CATEGORICAL = [
    "active_metal",
    "promoter",
    "support_material",
    "synthesis_method",
]
