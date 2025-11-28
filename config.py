import os

# Absolute path to the repository root (two levels above src/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths to data folders (relative to repo root)
DATA_DIR = os.path.join(ROOT_DIR, "notebooks", "data")
DRM_DIR = os.path.join(DATA_DIR, "drm")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Ensure processed folder exists
os.makedirs(PROCESSED_DIR, exist_ok=True)

TARGET = "ch4_conversion"

CATALYST_CATEGORICAL = [
    "active_metal",
    "promoter",
    "support_material",
    "synthesis_method",
]
