import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from config import DRM_DIR, PROCESSED_DIR

def run_week2_preprocessing():
    df = pd.read_csv(DRM_DIR + r"\drm_catalyst_performance.csv")

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    df.to_csv(PROCESSED_DIR + r"\drm_clean_processed.csv", index=False)

    print("Week 2 Preprocessing Complete → drm_clean_processed.csv")

if __name__ == "__main__":
    run_week2_preprocessing()
