import os

# Absolute path to repository root no matter where code is executed
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
