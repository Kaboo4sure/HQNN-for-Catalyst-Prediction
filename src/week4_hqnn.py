"""
Week 4 — HQNN-style Neural Model for DRM Catalyst Performance
Author: Taofeek Sanyaolu

This trains a PyTorch MLP on the clean DRM dataset as a stand-in for HQNN.
Later, you can replace the middle layer with a PennyLane quantum circuit.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import TARGET, CATALYST_CATEGORICAL

# Adjust if your config paths differ
BASE_DIR = r"C:\MachineLearning\HQNN\HQNN-for-Catalyst-Prediction\notebooks\data\drm"
CLEAN_FILE = "drm_catalyst_performance.csv"


# -------------------------------------------------------
# 1. Data loading + preprocessing (sklearn)
# -------------------------------------------------------
def load_clean_drm() -> pd.DataFrame:
    path = os.path.join(BASE_DIR, CLEAN_FILE)
    df = pd.read_csv(path)
    print(f"Loaded clean DRM data from {path}, shape = {df.shape}")
    return df


def build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, list, list]:
    cat_cols = CATALYST_CATEGORICAL
    num_cols = [c for c in df.columns if c not in cat_cols + [TARGET]]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    return preprocessor, num_cols, cat_cols


def preprocess_to_arrays(df: pd.DataFrame):
    df = df.copy()
    df = df.dropna(subset=[TARGET])  # just in case

    y = df[TARGET].values.astype(np.float32)

    preprocessor, num_cols, cat_cols = build_preprocessor(df)
    X = preprocessor.fit_transform(df.drop(columns=[TARGET]))

    X = X.astype(np.float32)

    print(f"Preprocessed X shape: {X.shape}, y shape: {y.shape}")
    return X, y, preprocessor


# -------------------------------------------------------
# 2. PyTorch Dataset / Model
# -------------------------------------------------------
class DRMTabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).view(-1, 1)  # (N, 1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class HQNNBaseline(nn.Module):
    """
    HQNN-style baseline:
      Input -> Dense -> (placeholder for quantum/hybrid layer) -> Dense -> Output
    For now this is a purely classical MLP. Later:
      - Replace `self.middle` with a quantum layer via PennyLane.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()

        # 🔽 This block can later be replaced with a quantum layer
        self.middle = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()

        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.middle(x)
        x = self.act2(x)
        x = self.fc_out(x)
        return x


# -------------------------------------------------------
# 3. Training loop
# -------------------------------------------------------
def train_hqnn(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_epochs: int = 50,
    hidden_dim: int = 128,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_ds = DRMTabularDataset(X_train, y_train)
    val_ds = DRMTabularDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = HQNNBaseline(input_dim=X.shape[1], hidden_dim=hidden_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training on device: {device}")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= len(train_ds)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
        val_loss /= len(val_ds)

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}")

    return model


# -------------------------------------------------------
# 4. Main Week 4 entrypoint
# -------------------------------------------------------
def run_week4_hqnn():
    df = load_clean_drm()
    X, y, preprocessor = preprocess_to_arrays(df)
    model = train_hqnn(X, y, num_epochs=50)

    # Save model + preprocessor for later analysis / SHAP
    out_dir = os.path.join(BASE_DIR, "models_week4")
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "hqnn_baseline.pt")
    torch.save(model.state_dict(), model_path)

    preproc_path = os.path.join(out_dir, "preprocessor.pkl")
    try:
        import joblib
        joblib.dump(preprocessor, preproc_path)
    except ImportError:
        print("joblib not installed, skipping preprocessor save.")

    print(f"\n✅ Week 4 HQNN-style model trained and saved to:\n  {model_path}")
    print(f"Preprocessor (if saved): {preproc_path}")


if __name__ == "__main__":
    run_week4_hqnn()
