import os
os.makedirs("outputs", exist_ok=True)



import shutil
import os
from config import DRM_DIR, PROCESSED_DIR

def run_week2b_consistency():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    files = [
        "drm_catalyst_performance.csv",
        "drm_corrupted_missing.csv",
        "drm_corrupted_noise.csv",
        "drm_corrupted_inconsistent.csv",
    ]

    for f in files:
        shutil.copy(DRM_DIR + "\\" + f, PROCESSED_DIR + "\\" + f)

    print("Week 2B → All corrupted datasets copied to processed folder.")

if __name__ == "__main__":
    run_week2b_consistency()
