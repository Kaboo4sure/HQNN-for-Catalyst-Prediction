"""
📘 Week 2B — Data Consistency Simulation
Author: Taofeek Sanyaolu
Purpose: Generate consistent and inconsistent datasets from preprocessed OC20 data
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1️⃣ Load Clean Processed Dataset
# ---------------------------------------------------------------
data_dir = r"data/processed"
base_path = os.path.join(data_dir, "oc20_train.csv")

df = pd.read_csv(base_path)
print(f"✅ Loaded {len(df)} clean samples from {base_path}")

features = ["natoms", "energy_init", "energy_relaxed", "energy_diff",
            "force_mean", "force_std", "volume", "atomic_number_mean", "atomic_number_std"]

# ---------------------------------------------------------------
# 2️⃣ Create Consistent and Inconsistent Copies
# ---------------------------------------------------------------
consistent_df = df.copy()

inconsistent_df = df.copy()

# --- Add controlled inconsistencies ---
# (a) Random noise to simulate sensor/measurement drift
for col in features:
    noise = np.random.normal(0, 0.05, size=len(inconsistent_df))  # 5% Gaussian noise
    inconsistent_df[col] = inconsistent_df[col] * (1 + noise)

# (b) Corrupt a small subset of energy readings
for col in ["energy_init", "energy_relaxed"]:
    corruption_idx = np.random.choice(len(inconsistent_df), size=int(0.02 * len(inconsistent_df)), replace=False)
    inconsistent_df.loc[corruption_idx, col] = inconsistent_df.loc[corruption_idx, col] * np.random.uniform(1.5, 2.0)

# (c) Shuffle atomic number mean to simulate label mix-ups
swap_idx = np.random.choice(len(inconsistent_df), size=int(0.01 * len(inconsistent_df)), replace=False)
inconsistent_df.loc[swap_idx, "atomic_number_mean"] = inconsistent_df["atomic_number_mean"].sample(frac=1.0).values

# Label sources
consistent_df["source"] = "consistent"
inconsistent_df["source"] = "inconsistent"

# Combine datasets
merged_df = pd.concat([consistent_df, inconsistent_df], ignore_index=True)
print(f"✅ Created merged dataset: {len(merged_df)} rows total "
      f"({len(consistent_df)} consistent + {len(inconsistent_df)} inconsistent)")

# ---------------------------------------------------------------
# 3️⃣ Save Datasets
# ---------------------------------------------------------------
os.makedirs(data_dir, exist_ok=True)
consistent_df.to_csv(os.path.join(data_dir, "oc20_consistent.csv"), index=False)
inconsistent_df.to_csv(os.path.join(data_dir, "oc20_inconsistent.csv"), index=False)
merged_df.to_csv(os.path.join(data_dir, "oc20_merged_sources.csv"), index=False)
print("💾 Saved consistent, inconsistent, and merged datasets in data/processed")

# ---------------------------------------------------------------
# 4️⃣ Visualize for Quick Comparison
# ---------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.kdeplot(data=merged_df, x="energy_diff", hue="source", fill=True, common_norm=False, alpha=0.5)
plt.title("Energy Difference Distribution — Consistent vs Inconsistent Sources")
plt.xlabel("Energy Difference (eV)")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

print("✅ Simulation complete — ready for Week 3 training.")
