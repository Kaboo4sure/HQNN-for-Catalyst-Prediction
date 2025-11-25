import os
import pandas as pd
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

    # Remove ID and irrelevant output columns
    drop_cols = ["sample_id", "co2_conversion", "h2_co_ratio", "catalyst_stability"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

    # Target variable
    y = df["ch4_conversion"].astype(float).values
    X = df.drop(columns=["ch4_conversion"])

    # Encode categorical columns
    categorical_cols = [
        "active_metal",
        "promoter",
        "support_material",
        "synthesis_method",
    ]
    for col in categorical_cols:
        if col in X.columns:
            X[col] = pd.Categorical(X[col]).codes

    # Impute missing numerical values
    if X.isnull().sum().sum() > 0:
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # ---- FIX: Drop rows where target y is NaN ----
    mask = ~np.isnan(y)
    X = X[mask]
    y = y[mask]
    # ---------------------------------------------

    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, scaler

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
