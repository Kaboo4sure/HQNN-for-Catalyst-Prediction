import os
os.makedirs("outputs", exist_ok=True)


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
    plt.savefig("outputs/week1_corr_heatmap.png", dpi=300)


if __name__ == "__main__":
    run_week1_exploration()
