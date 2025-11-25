"""
Week 5 — Explainability:
- SHAP feature importance (XGBoost)
- PCA visualization of dataset drift

Run alone:
    python src/week5_explainability.py

Run in full pipeline:
    python main.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor
from config import DRM_DIR


# ---------------------------------------------------------
# 1. Load datasets
# ---------------------------------------------------------
def load_drm_datasets():
    files = {
        "Clean Data": "drm_catalyst_performance.csv",
        "Missing Data": "drm_corrupted_missing.csv",
        "Noise Data": "drm_corrupted_noise.csv",
        "Inconsistent Data": "drm_corrupted_inconsistent.csv",
    }

    datasets = {}
    print("\nWEEK 5 — DATASET OVERVIEW")
    print("=" * 60)

    for name, fname in files.items():
        path = os.path.join(DRM_DIR, fname)
        df = pd.read_csv(path)
        datasets[name] = df

        print(f"{name:18} — {df.shape[0]} samples, {df.shape[1]} columns, "
              f"{df.isnull().sum().sum()} missing values")

    return datasets


# ---------------------------------------------------------
# 2. Basic preprocessing for XGBoost
# ---------------------------------------------------------
def prepare_tabular_data(df):
    df = df.copy()

    drop_cols = ["sample_id", "co2_conversion", "h2_co_ratio", "catalyst_stability"]
    for col in drop_cols:
        if col in df:
            df = df.drop(columns=[col])

    y = df["ch4_conversion"].values.astype(np.float32)
    X = df.drop(columns=["ch4_conversion"])

    # Encode categoricals
    cat_cols = ["active_metal", "promoter", "support_material", "synthesis_method"]
    for col in cat_cols:
        if col in X:
            X[col] = pd.Categorical(X[col]).codes

    # Impute features
    if X.isnull().sum().sum() > 0:
        imp = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    # Impute target
    if np.isnan(y).sum() > 0:
        print("⚠ Imputing missing target values with mean")
        imp_y = SimpleImputer(strategy="mean")
        y = imp_y.fit_transform(y.reshape(-1, 1)).flatten()

    return X.values, y, list(X.columns)


# ---------------------------------------------------------
# 3. Train XGBoost baselines
# ---------------------------------------------------------
def train_xgboost(datasets):
    print("\nTRAINING XGBOOST BASELINES")
    print("=" * 60)

    results = {}
    models = {}
    splits = {}

    for name, df in datasets.items():
        print(f"\n▶ Dataset: {name}")

        X, y, feature_names = prepare_tabular_data(df)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )
        model.fit(X_train_s, y_train)

        preds = model.predict(X_test_s)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        print(f"   MAE: {mae:.2f} | R²: {r2:.3f}")

        results[name] = {"MAE": mae, "R2": r2}
        models[name] = model
        splits[name] = {
            "X_train": X_train_s,
            "X_test": X_test_s,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": feature_names,
        }

    return results, models, splits


# ---------------------------------------------------------
# 4. SHAP explainability
# ---------------------------------------------------------
def run_shap(models, splits):
    print("\nWEEK 5 — SHAP EXPLAINABILITY")
    print("=" * 60)

    shap_results = {}

    for name, model in models.items():
        print(f"\n▶ SHAP for {name}")

        X_train = splits[name]["X_train"]
        X_test = splits[name]["X_test"]
        feature_names = splits[name]["feature_names"]

        background = X_train[:200]  # SHAP background
        explainer = shap.TreeExplainer(model, data=background)
        shap_values = explainer.shap_values(X_test)

        mean_abs = np.abs(shap_values).mean(axis=0)
        shap_results[name] = {
            "shap_values": shap_values,
            "mean_abs": mean_abs,
            "features": feature_names,
        }

        # Print top-5 features
        idx = np.argsort(-mean_abs)
        print("   Top 5 features:")
        for i in range(5):
            print(f"     {i+1}. {feature_names[idx[i]]}: {mean_abs[idx[i]]:.4f}")

        # Plot importance
        plt.figure(figsize=(8, 4))
        plt.barh(feature_names[::-1], mean_abs[::-1])
        plt.title(f"SHAP Global Importance — {name}")
        plt.tight_layout()
        plt.show()

    return shap_results


# ---------------------------------------------------------
# 5. PCA visualization across datasets
# ---------------------------------------------------------
def run_pca(datasets):
    print("\nWEEK 5 — PCA VISUALIZATION")
    print("=" * 60)

    frames = []
    for name, df in datasets.items():
        temp = df.copy()
        temp["__dataset__"] = name
        frames.append(temp)
    all_df = pd.concat(frames)

    labels = all_df["__dataset__"].values
    all_df = all_df.drop(columns=["__dataset__", "sample_id", "co2_conversion",
                                  "h2_co_ratio", "catalyst_stability"],
                         errors="ignore")

    if "ch4_conversion" in all_df:
        all_df = all_df.drop(columns=["ch4_conversion"])

    # Encode categoricals
    for col in ["active_metal", "promoter", "support_material", "synthesis_method"]:
        if col in all_df:
            all_df[col] = pd.Categorical(all_df[col]).codes

    # Impute
    imp = SimpleImputer(strategy="mean")
    X = imp.fit_transform(all_df)

    # Scale
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(Xs)

    plt.figure(figsize=(8, 6))
    for name in np.unique(labels):
        mask = labels == name
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, alpha=0.6)

    plt.title("PCA: Clean vs Corrupted Data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return X_pca, labels


# ---------------------------------------------------------
# 6. Week 5 wrapper
# ---------------------------------------------------------
def run_week5_explainability():
    print("\n" + "=" * 60)
    print("       WEEK 5 — SHAP & PCA Explainability")
    print("=" * 60)

    datasets = load_drm_datasets()
    baseline, models, splits = train_xgboost(datasets)
    shap_results = run_shap(models, splits)
    pca_results = run_pca(datasets)

    print("\nWEEK 5 COMPLETE ✓")
    print("• XGBoost baselines trained")
    print("• SHAP global importance computed")
    print("• PCA drift visualized\n")

    return baseline, shap_results, pca_results


if __name__ == "__main__":
    run_week5_explainability()
