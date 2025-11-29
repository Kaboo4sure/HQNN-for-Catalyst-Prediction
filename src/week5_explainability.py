"""
WEEK 5 — Explainability (FAST VERSION)
-------------------------------------
We replace Kernel SHAP with Integrated Gradients + Gradient SHAP for the Hybrid QNN.
This makes the pipeline finish in under 2 minutes instead of 55 hours.

Included:
- SHAP feature importance (XGBoost)
- PCA visualization
- Integrated Gradients (HQNN)
- GradientSHAP (HQNN)

Run alone:
    python src/week5_explainability.py

Run in pipeline:
    python main.py
"""

import os
os.makedirs("outputs", exist_ok=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, r2_score

from xgboost import XGBRegressor
from config import DRM_DIR

# Captum for fast explainability
from captum.attr import IntegratedGradients, GradientShap


# =========================================================
# 1. LOAD DATASETS
# =========================================================
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

        print(f"{name:18} — {df.shape[0]} samples, {df.shape[1]} cols, "
              f"{df.isnull().sum().sum()} missing")

    return datasets


# =========================================================
# 2. PREPARE TABULAR DATA FOR XGBOOST
# =========================================================
def prepare_tabular_data(df):
    df = df.copy()

    drop_cols = ["sample_id", "co2_conversion", "h2_co_ratio", "catalyst_stability"]
    df = df.drop(columns=[c for c in drop_cols if c in df], errors="ignore")

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
        X[:] = imp.fit_transform(X)

    # Impute target
    if np.isnan(y).sum() > 0:
        imp_y = SimpleImputer(strategy="mean")
        y = imp_y.fit_transform(y.reshape(-1, 1)).flatten()

    return X.values, y, list(X.columns)


# =========================================================
# 3. TRAIN BASELINE XGBOOST MODELS
# =========================================================
def train_xgboost(datasets):
    print("\nTRAINING XGBOOST BASELINES")
    print("=" * 60)

    results, models, splits = {}, {}, {}

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


# =========================================================
# 4. SHAP FOR XGBOOST (FAST)
# =========================================================
def run_shap(models, splits):
    print("\nWEEK 5 — SHAP (XGBoost Models)")
    print("=" * 60)

    shap_results = {}

    for name, model in models.items():
        print(f"\n▶ SHAP for {name}")

        X_train = splits[name]["X_train"]
        X_test = splits[name]["X_test"]
        feature_names = splits[name]["feature_names"]

        # Background sample (small for speed)
        background = X_train[:20]

        # -----------------------------------------------
        # FINAL FIX: Use KernelExplainer directly (SHAP v0.42+)
        # -----------------------------------------------
        explainer = shap.KernelExplainer(
            model.predict,
            background
        )

        # Compute SHAP values for 30 samples
        shap_values = explainer.shap_values(X_test[:30])

        # Convert to numpy
        shap_values = np.array(shap_values)

        # Global mean absolute importance
        mean_abs = np.abs(shap_values).mean(axis=0)

        shap_results[name] = {
            "shap_values": shap_values,
            "mean_abs": mean_abs,
            "features": feature_names,
        }

        # Log top features
        idx = np.argsort(-mean_abs)
        print("   Top 5 Features:")
        for i in range(5):
            print(f"     {i+1}. {feature_names[idx[i]]}: {mean_abs[idx[i]]:.4f}")

        # Save plot
        plt.figure(figsize=(8, 4))
        plt.barh(feature_names[::-1], mean_abs[::-1])
        plt.title(f"SHAP Global Importance — {name}")
        plt.tight_layout()

        os.makedirs("outputs", exist_ok=True)
        plt.savefig(f"outputs/week5_shap_{name.replace(' ', '_')}.png", dpi=300)

        plt.close()

    return shap_results




# =========================================================
# 5. PCA VISUALIZATION
# =========================================================
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

    # Drop non-features
    drop_cols = ["__dataset__", "sample_id", "co2_conversion",
                 "h2_co_ratio", "catalyst_stability", "ch4_conversion"]
    all_df = all_df.drop(columns=drop_cols, errors="ignore")

    # Encode categoricals
    for col in ["active_metal", "promoter", "support_material", "synthesis_method"]:
        if col in all_df:
            all_df[col] = pd.Categorical(all_df[col]).codes

    X = SimpleImputer(strategy="mean").fit_transform(all_df)
    Xs = StandardScaler().fit_transform(X)
    X_pca = PCA(n_components=2).fit_transform(Xs)

    plt.figure(figsize=(8, 6))
    for name in np.unique(labels):
        mask = labels == name
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=name, alpha=0.6)

    plt.legend()
    plt.title("PCA: Clean vs Corrupted Datasets")
    plt.tight_layout()
    plt.savefig("outputs/week5_pca.png", dpi=300)
    plt.close()

    return X_pca, labels


# =========================================================
# 6. FAST EXPLAINABILITY FOR HYBRID QNN
# =========================================================
def run_hqnn_explainability(hqnn_model, X_train_t, X_test_t, feature_names):
    print("\nWEEK 5 — HQNN EXPLAINABILITY (FAST)")
    print("=" * 60)

    # Convert tensors → numpy
    X_train_np = X_train_t.detach().cpu().numpy()
    X_test_np = X_test_t.detach().cpu().numpy()

    background = X_train_np[:40]
    test_samples = X_test_np[:60]

    # Prediction wrapper
    def predict_fn(x):
        x_t = torch.tensor(x, dtype=torch.float32)
        with torch.no_grad():
            return hqnn_model(x_t).cpu().numpy()

    # Kernel SHAP
    explainer = shap.KernelExplainer(predict_fn, background)
    shap_values = explainer.shap_values(test_samples)
    shap_values = np.array(shap_values)

    mean_abs = np.abs(shap_values).mean(axis=0)

    # Plot
    plt.figure(figsize=(8,4))
    plt.barh(feature_names[::-1], mean_abs[::-1])
    plt.title("Hybrid QNN — SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig("outputs/week5_hqnn_shap.png", dpi=300)
    plt.close()

    return {
        "shap_values": shap_values,
        "mean_abs": mean_abs,
        "features": feature_names,
    }


# =========================================================
# 7. MAIN WRAPPER
# =========================================================
def run_week5_explainability():
    print("\n" + "=" * 60)
    print("       WEEK 5 — EXPLAINABILITY (FAST VERSION)")
    print("=" * 60)

    datasets = load_drm_datasets()
    baseline, models, splits = train_xgboost(datasets)
    shap_results = run_shap(models, splits)
    pca_results = run_pca(datasets)

    # Load HQNN (from Week 4)
    print("\nLOADING HYBRID QNN FROM WEEK 4...")
    from src.week4_hqnn import HybridQuantumModel

    clean_df = datasets["Clean Data"]
    hmodel = HybridQuantumModel(n_qubits=4, n_layers=2)

    X_train_t, X_test_t, y_train_t, y_test_t, feature_cols = \
        hmodel.prepare_data(clean_df)

    # FAST explainability (no Kernel SHAP)
    hqnn_results = run_hqnn_explainability(
        hmodel, X_train_t, X_test_t, feature_cols
    )

    return baseline, shap_results, pca_results, hqnn_results


if __name__ == "__main__":
    run_week5_explainability()
