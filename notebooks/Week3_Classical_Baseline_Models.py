"""
🧠 Week 3 — Classical Baseline Models
Project: HQNN for Catalyst Performance Prediction
Author: Taofeek Sanyaolu

Goal:
Compare model performance on:
  1) Clean 100-sample dataset
  2) Manually edited inconsistent 100-sample dataset
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ---------------------------------------------------------------
# 1️⃣ Paths & Config
# ---------------------------------------------------------------
data_dir = r"data/processed"

clean_path = os.path.join(data_dir, "oc20_first100_clean.csv")
inconsistent_path = os.path.join(data_dir, "oc20_first100_inconsistent_manual.csv")

print("✅ Using files:")
print("  Clean:        ", clean_path)
print("  Inconsistent: ", inconsistent_path)

# Same feature set as Week 2
FEATURES = [
    "natoms",
    "energy_init",
    "energy_relaxed",
    "energy_diff",
    "force_mean",
    "force_std",
    "volume",
    "atomic_number_mean",
    "atomic_number_std",
]

TARGET = "energy_diff"   # we will predict this

# ---------------------------------------------------------------
# 2️⃣ Load Datasets
# ---------------------------------------------------------------
clean_df = pd.read_csv(clean_path)
incons_df = pd.read_csv(inconsistent_path)

print(f"\nClean dataset shape:        {clean_df.shape}")
print(f"Inconsistent dataset shape: {incons_df.shape}")

# Basic sanity checks
assert all(col in clean_df.columns for col in FEATURES), "Missing features in clean_df!"
assert all(col in incons_df.columns for col in FEATURES), "Missing features in incons_df!"

# ---------------------------------------------------------------
# 3️⃣ Helper: Train & Evaluate on a Dataset
# ---------------------------------------------------------------
def train_and_evaluate(df, label):
    """
    Train Linear Regression and Random Forest on given dataframe.
    Returns metrics dict for both models.
    """
    print(f"\n🧪 Training on {label} dataset")

    X = df[FEATURES].values
    y = df[TARGET].values

    # Use 80/20 train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standard scaling (fit on train, apply to test)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ----- Model 1: Linear Regression -----
    linreg = LinearRegression()
    linreg.fit(X_train_scaled, y_train)

    y_pred_lr = linreg.predict(X_test_scaled)

    results["LinearRegression"] = {
        "MSE": mean_squared_error(y_test, y_pred_lr),
        "MAE": mean_absolute_error(y_test, y_pred_lr),
        "R2": r2_score(y_test, y_pred_lr),
    }

    # ----- Model 2: Random Forest -----
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )
    rf.fit(X_train, y_train)  # RF can work well on unscaled data

    y_pred_rf = rf.predict(X_test)

    results["RandomForest"] = {
        "MSE": mean_squared_error(y_test, y_pred_rf),
        "MAE": mean_absolute_error(y_test, y_pred_rf),
        "R2": r2_score(y_test, y_pred_rf),
    }

    # ----- Quick scatter comparison for RF (better nonlinear fit) -----
    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_pred_rf, alpha=0.7)
    lims = [min(y_test.min(), y_pred_rf.min()), max(y_test.max(), y_pred_rf.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("True energy_diff")
    plt.ylabel("Predicted energy_diff (RF)")
    plt.title(f"Random Forest: True vs Predicted ({label})")
    plt.tight_layout()
    plt.show()

    return results

# ---------------------------------------------------------------
# 4️⃣ Run on Clean & Inconsistent Datasets
# ---------------------------------------------------------------
clean_results = train_and_evaluate(clean_df, "CLEAN (first 100)")
incons_results = train_and_evaluate(incons_df, "INCONSISTENT (manual)")

# ---------------------------------------------------------------
# 5️⃣ Compare Metrics Side-by-Side
# ---------------------------------------------------------------
def print_results_table(clean_res, incons_res):
    models = clean_res.keys()
    rows = []
    for m in models:
        rows.append({
            "Model": m,
            "Dataset": "Clean",
            "MSE": clean_res[m]["MSE"],
            "MAE": clean_res[m]["MAE"],
            "R2": clean_res[m]["R2"],
        })
        rows.append({
            "Model": m,
            "Dataset": "Inconsistent",
            "MSE": incons_res[m]["MSE"],
            "MAE": incons_res[m]["MAE"],
            "R2": incons_res[m]["R2"],
        })
    res_df = pd.DataFrame(rows)
    print("\n📊 Performance Comparison (lower MSE/MAE is better, higher R2 is better):")
    print(res_df.to_string(index=False))

    # Optional: barplot for R2
    plt.figure(figsize=(8, 5))
    sns.barplot(data=res_df, x="Model", y="R2", hue="Dataset")
    plt.title("R² Comparison — Clean vs Inconsistent")
    plt.tight_layout()
    plt.show()

print_results_table(clean_results, incons_results)

print("\n✅ Week 3 classical baselines complete.")
print("Next: we can add HQNN / hybrid QNN in a similar framework.")
