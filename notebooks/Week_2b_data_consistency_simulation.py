"""
📘 Week 2B — Manual Data Consistency Simulation (100 Samples Only)
Author: Taofeek Sanyaolu
Purpose: Generate a clean 100-sample dataset for manual editing.
"""

import os
import pandas as pd

# ---------------------------------------------------------------
# 1️⃣ Load Clean Processed Dataset (FIRST 100 ONLY)
# ---------------------------------------------------------------
data_dir = r"data/processed"
base_path = os.path.join(data_dir, "oc20_train.csv")

df = pd.read_csv(base_path)

# Keep only the first 100 rows
df_100 = df.head(100)
print(f"✅ Loaded {len(df_100)} samples (first 100 rows only)")

# ---------------------------------------------------------------
# 2️⃣ Save the Clean Editable Version
# ---------------------------------------------------------------
clean_path = os.path.join(data_dir, "oc20_first100_clean.csv")
df_100.to_csv(clean_path, index=False)

print(f"💾 Saved clean editable dataset → {clean_path}")
print("📝 You should now manually create inconsistent versions of this file.")
print("\nSuggested manual edits include:")
print("• Change energy values (simulate measurement error)")
print("• Change natoms (simulate labeling mistakes)")
print("• Edit atomic_number_mean / std (simulate wrong compositions)")
print("• Modify volume or forces (simulate experimental drift)")
print("\nAfter editing, save your file as:")
print("oc20_first100_inconsistent_manual.csv")

# ---------------------------------------------------------------
# 3️⃣ Visualization of the clean dataset (optional)
# ---------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(8,5))
    sns.kdeplot(df_100["energy_diff"], fill=True, color="blue")
    plt.title("Energy Difference Distribution — First 100 Clean Samples")
    plt.xlabel("Energy Diff (eV)")
    plt.tight_layout()
    plt.show()

except ImportError:
    print("Visualization skipped (matplotlib/seaborn not installed).")

print("\n🎉 Week 2B is complete — manually edit your dataset before Week 3.")
