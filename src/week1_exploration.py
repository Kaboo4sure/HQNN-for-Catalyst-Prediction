import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from config import DRM_DIR

def run_week1_exploration():
    df = pd.read_csv(DRM_DIR + r"\drm_catalyst_performance.csv")

    print(df.head())
    print(df.describe())

    plt.figure(figsize=(6, 4))
    sns.histplot(df["ch4_conversion"], kde=True)
    plt.title("CH₄ Conversion Distribution")
    plt.show()

if __name__ == "__main__":
    run_week1_exploration()
