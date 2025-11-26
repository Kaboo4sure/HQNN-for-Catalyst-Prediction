import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


import os
import pandas as pd
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer


from config import PROCESSED_DIR, TARGET, CATALYST_CATEGORICAL

def preprocess(df):
    df = df.copy()

    # Remove non-features
    drop_cols = ["sample_id", "co2_conversion", "h2_co_ratio", "catalyst_stability"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Extract y
    y = df["ch4_conversion"].astype(float).values

    # Extract X
    X = df.drop(columns=["ch4_conversion"])

    # Encode categoricals
    categorical_cols = ["active_metal", "promoter", "support_material", "synthesis_method"]
    for col in categorical_cols:
        if col in X.columns:
            X[col] = pd.Categorical(X[col]).codes

    # Impute missing in X
    if X.isnull().sum().sum() > 0:
        imp = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imp.fit_transform(X), columns=X.columns)

    # FIX: remove rows where y is NaN
    mask = ~np.isnan(y)
    X = X.iloc[mask]
    y = y[mask]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler


def evaluate(df):
    X, y, _ = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    return {
        "MSE": mean_squared_error(y_test, preds),
        "R2": r2_score(y_test, preds),
    }

def run_week3_models():
    datasets = {
        "Clean": "drm_catalyst_performance.csv",
        "Missing": "drm_corrupted_missing.csv",
        "Noise": "drm_corrupted_noise.csv",
        "Inconsistent": "drm_corrupted_inconsistent.csv",
    }

    results = {}

    for name, file in datasets.items():
        df = pd.read_csv(PROCESSED_DIR + "\\" + file)
        results[name] = evaluate(df)

    print("\n=== Week 3 Results ===")
    for k, v in results.items():
        print(k, v)

if __name__ == "__main__":
    run_week3_models()
