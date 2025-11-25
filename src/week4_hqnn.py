"""
Week 4 — Hybrid Quantum-Classical Model + Classical Baselines
Replicates Colab Steps 3–6 for DRM catalyst data.

Datasets:
- drm_catalyst_performance.csv        (clean)
- drm_corrupted_missing.csv           (15% missing values)
- drm_corrupted_noise.csv             (10% Gaussian noise)
- drm_corrupted_inconsistent.csv      (mixed units / inconsistencies)

This script:
  1) Loads all four datasets.
  2) Trains HybridQuantumModel on each dataset.
  3) Trains Random Forest, MLP, XGBoost on each dataset.
  4) Prints MAE / R² and plots comparisons.

Run via:
  python .\src\week4_hqnn.py
or via main pipeline:
  python .\main.py
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pennylane as qml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.impute import SimpleImputer

from config import DRM_DIR

warnings.filterwarnings("ignore")


# -------------------------------------------------------------------
# 1. Load DRM datasets (maps Colab Step 3)
# -------------------------------------------------------------------
def load_drm_datasets():
    files = {
        "Clean Data": "drm_catalyst_performance.csv",
        "Missing Data": "drm_corrupted_missing.csv",
        "Noise Data": "drm_corrupted_noise.csv",
        "Inconsistent Data": "drm_corrupted_inconsistent.csv",
    }

    datasets = {}
    print("DATASET OVERVIEW")
    print("=" * 50)

    for name, fname in files.items():
        path = os.path.join(DRM_DIR, fname)
        df = pd.read_csv(path)
        datasets[name] = df

        if name == "Missing Data":
            missing_count = df.isnull().sum().sum()
            print(f"{name:18}: {df.shape[0]} samples, {df.shape[1]} features, {missing_count} missing values")
        else:
            print(f"{name:18}: {df.shape[0]} samples, {df.shape[1]} features")

    clean_data = datasets["Clean Data"]
    print("\nTARGET VARIABLE (CH4 Conversion)")
    print("=" * 50)
    print(f"Clean data - Mean: {clean_data['ch4_conversion'].mean():.1f}%, "
          f"Std: {clean_data['ch4_conversion'].std():.1f}%")
    print(f"Range: {clean_data['ch4_conversion'].min():.1f}% to "
          f"{clean_data['ch4_conversion'].max():.1f}%")

    return datasets


# -------------------------------------------------------------------
# 2. Hybrid Quantum-Classical Model (maps Colab Step 4)
# -------------------------------------------------------------------
class HybridQuantumModel:
    def __init__(self, n_qubits=4, n_layers=2):
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.scaler = StandardScaler()

        # Quantum device (simulator)
        self.dev = qml.device("default.qubit", wires=n_qubits)

        # Define quantum circuit
        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            # Encode classical data into quantum states
            for i in range(n_qubits):
                qml.RY(inputs[i] * np.pi, wires=i)

            # Variational quantum layers
            for layer in range(n_layers):
                # Single-qubit rotations
                for i in range(n_qubits):
                    qml.Rot(
                        weights[layer, i, 0],
                        weights[layer, i, 1],
                        weights[layer, i, 2],
                        wires=i,
                    )

                # Entangling layer
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i + 1])
                if n_qubits > 1:
                    qml.CNOT(wires=[n_qubits - 1, 0])

            # Measurement (one expval per qubit)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.quantum_circuit = quantum_circuit

        # Initialize quantum weights
        self.weights = nn.Parameter(torch.randn(n_layers, n_qubits, 3))

        # Classical neural network (post-quantum)
        self.classical_nn = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),  # Predict CH4 conversion
        )

    def prepare_data(self, dataframe, target_col="ch4_conversion"):
        """Prepare data for training (matches Colab's prepare_data)."""
        df = dataframe.copy()

        # Select features (exclude IDs and secondary outputs)
        exclude_cols = [
            "sample_id",
            "co2_conversion",
            "h2_co_ratio",
            "catalyst_stability",
        ]
        feature_cols = [
            col
            for col in df.columns
            if col not in exclude_cols and col != target_col
        ]

        X = df[feature_cols].copy()

        # Handle categorical variables
        categorical_cols = [
            "active_metal",
            "promoter",
            "support_material",
            "synthesis_method",
        ]
        for col in categorical_cols:
            if col in X.columns:
                X[col] = pd.Categorical(X[col]).codes

        # Handle missing values by imputation
        if X.isnull().sum().sum() > 0:
            imputer = SimpleImputer(strategy="mean")
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        y = df[target_col].values.astype(np.float32)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return (
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(X_test_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
            feature_cols,
        )

    def forward(self, x):
        """Hybrid forward pass: quantum + classical."""
        # Use first n_qubits features for quantum processing
        x_quantum = x[:, : self.n_qubits]

        # Quantum processing per sample
        quantum_outputs = []
        for i in range(x.shape[0]):
            q_out = self.quantum_circuit(x_quantum[i], self.weights)
            quantum_outputs.append(q_out)

        quantum_tensor = torch.stack(quantum_outputs)

        # Classical processing
        output = self.classical_nn(quantum_tensor)

        return output.squeeze()

    def train_model(self, X_train, y_train, epochs=200, lr=0.01):
        """Train the hybrid model (matches Colab)."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = self.forward(X_train)
            loss = criterion(predictions, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if epoch % 40 == 0:
                print(f"      Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f}")

        return losses

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        with torch.no_grad():
            predictions = self.forward(X_test)
            mae = mean_absolute_error(y_test.numpy(), predictions.numpy())
            r2 = r2_score(y_test.numpy(), predictions.numpy())

        return mae, r2, predictions.numpy()

    def parameters(self):
        """Get all trainable parameters."""
        return [self.weights] + list(self.classical_nn.parameters())


# -------------------------------------------------------------------
# 3. Train Hybrid QNN on all datasets (maps Colab Step 5)
# -------------------------------------------------------------------
def run_hybrid_qnn_experiments(datasets):
    print("\nTRAINING HYBRID QUANTUM-CLASSICAL MODELS")
    print("=" * 60)

    results = {}

    # Reuse one instance only for data prep (as in Colab),
    # but create a new model for each dataset.
    prep_model = HybridQuantumModel(n_qubits=4, n_layers=2)
    print("\nInitializing Hybrid Quantum-Classical Model for preprocessing...")
    print("√ Model initialized successfully!\n")

    for name, data in datasets.items():
        print(f"\n▶ Training on {name}...")

        # Prepare data
        X_train, X_test, y_train, y_test, features = prep_model.prepare_data(data)
        print(
            f"   Features: {len(features)} | "
            f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}"
        )

        # Fresh model per dataset
        model = HybridQuantumModel(n_qubits=4, n_layers=2)

        # Train model, I resuded epochs from 200 to 40
        losses = model.train_model(X_train, y_train, epochs=40, lr=0.01)

        # Evaluate
        mae, r2, predictions = model.evaluate(X_test, y_test)

        results[name] = {
            "MAE": mae,
            "R2": r2,
            "predictions": predictions,
            "true_values": y_test.numpy(),
        }

        print(f"   √ {name} - MAE: {mae:.2f}% | R²: {r2:.3f}")

    print("\n" + "=" * 60)
    print("HYBRID QNN PERFORMANCE SUMMARY")
    print("=" * 60)
    for name, res in results.items():
        print(f"{name:20} | MAE: {res['MAE']:6.2f}% | R²: {res['R2']:6.3f}")

    # Visualization: MAE, R², Predictions vs Actual
    plt.figure(figsize=(15, 5))

    # MAE comparison
    plt.subplot(1, 3, 1)
    mae_values = [results[name]["MAE"] for name in datasets.keys()]
    plt.bar(datasets.keys(), mae_values, color=["green", "red", "orange", "purple"])
    plt.ylabel("MAE (%)")
    plt.title("Hybrid QNN Performance (MAE)")
    plt.xticks(rotation=45)
    for i, v in enumerate(mae_values):
        plt.text(i, v + 0.1, f"{v:.1f}%", ha="center")

    # R² comparison
    plt.subplot(1, 3, 2)
    r2_values = [results[name]["R2"] for name in datasets.keys()]
    plt.bar(datasets.keys(), r2_values, color=["green", "red", "orange", "purple"])
    plt.ylabel("R² Score")
    plt.title("Hybrid QNN Performance (R²)")
    plt.xticks(rotation=45)
    for i, v in enumerate(r2_values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")

    # Predictions vs Actual
    plt.subplot(1, 3, 3)
    for name, res in results.items():
        plt.scatter(
            res["true_values"],
            res["predictions"],
            alpha=0.6,
            label=name,
        )
    plt.plot([0, 100], [0, 100], "k--", alpha=0.5)
    plt.xlabel("Actual CH4 Conversion (%)")
    plt.ylabel("Predicted CH4 Conversion (%)")
    plt.title("Hybrid QNN – Predictions vs Actual")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Performance degradation analysis
    print("\nPERFORMANCE DEGRADATION (vs Clean Data, MAE-based)")
    clean_mae = results["Clean Data"]["MAE"]
    for name in ["Missing Data", "Noise Data", "Inconsistent Data"]:
        degradation = ((results[name]["MAE"] - clean_mae) / clean_mae) * 100
        print(f"{name:20} | Performance drop: {degradation:+.1f}%")

    return results


# -------------------------------------------------------------------
# 4. Classical models comparison (maps Colab Step 6)
# -------------------------------------------------------------------
def run_classical_models(datasets, hybrid_results):
    print("\nCOMPARING WITH CLASSICAL MODELS")
    print("=" * 50)

    classical_results = {}

    classical_models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "MLP": MLPRegressor(
            hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42
        ),
        "XGBoost": XGBRegressor(random_state=42),
    }

    for model_name, model in classical_models.items():
        print(f"\n▶ Training {model_name}...")
        model_results = {}

        for data_name, data in datasets.items():
            df = data.copy()
            exclude_cols = [
                "sample_id",
                "co2_conversion",
                "h2_co_ratio",
                "catalyst_stability",
            ]
            feature_cols = [
                col
                for col in df.columns
                if col not in exclude_cols and col != "ch4_conversion"
            ]

            X = df[feature_cols].copy()
            categorical_cols = [
                "active_metal",
                "promoter",
                "support_material",
                "synthesis_method",
            ]

            for col in categorical_cols:
                if col in X.columns:
                    X[col] = pd.Categorical(X[col]).codes

            # Handle missing
            if X.isnull().sum().sum() > 0:
                imputer = SimpleImputer(strategy="mean")
                X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

            y = df["ch4_conversion"].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            model_results[data_name] = {"MAE": mae, "R2": r2}

            print(
                f"   {data_name:20} | MAE: {mae:.2f}% | R²: {r2:.3f}"
            )

        classical_results[model_name] = model_results

    print("\n" + "=" * 60)
    print("FINAL COMPARISON: HYBRID QNN vs CLASSICAL MODELS")
    print("=" * 60)

    # Build comparison table
    comparison_data = []
    for data_name in datasets.keys():
        row = [data_name]

        # Hybrid results
        row.extend(
            [hybrid_results[data_name]["MAE"], hybrid_results[data_name]["R2"]]
        )

        # Classical
        for model_name in classical_models.keys():
            row.extend(
                [
                    classical_results[model_name][data_name]["MAE"],
                    classical_results[model_name][data_name]["R2"],
                ]
            )

        comparison_data.append(row)

    columns = ["Dataset", "QNN_MAE", "QNN_R2"]
    for model_name in classical_models.keys():
        columns.extend([f"{model_name}_MAE", f"{model_name}_R2"])

    comparison_df = pd.DataFrame(comparison_data, columns=columns)
    print(comparison_df.round(3))

    # Visual MAE / R² comparison
    plt.figure(figsize=(15, 8))
    models = ["Hybrid QNN"] + list(classical_models.keys())
    x_pos = np.arange(len(datasets))
    width = 0.15

    # MAE
    plt.subplot(2, 1, 1)
    for i, model in enumerate(models):
        if model == "Hybrid QNN":
            mae_values = [
                hybrid_results[name]["MAE"] for name in datasets.keys()
            ]
        else:
            mae_values = [
                classical_results[model][name]["MAE"]
                for name in datasets.keys()
            ]
        plt.bar(x_pos + i * width, mae_values, width, label=model)

    plt.xlabel("Dataset")
    plt.ylabel("MAE (%)")
    plt.title("Model Comparison - MAE across Different Data Quality")
    plt.xticks(x_pos + width * 1.5, datasets.keys())
    plt.legend()
    plt.grid(True, alpha=0.3)

    # R²
    plt.subplot(2, 1, 2)
    for i, model in enumerate(models):
        if model == "Hybrid QNN":
            r2_values = [
                hybrid_results[name]["R2"] for name in datasets.keys()
            ]
        else:
            r2_values = [
                classical_results[model][name]["R2"]
                for name in datasets.keys()
            ]
        plt.bar(x_pos + i * width, r2_values, width, label=model)

    plt.xlabel("Dataset")
    plt.ylabel("R² Score")
    plt.title("Model Comparison - R² across Different Data Quality")
    plt.xticks(x_pos + width * 1.5, datasets.keys())
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\nKEY INSIGHTS:")
    print("• Lower MAE and higher R² indicate better performance.")
    print("• Observe performance drop from Clean → corrupted datasets.")
    print("• You can now quantitatively compare Hybrid QNN vs RF / MLP / XGBoost robustness.")

    return classical_results, comparison_df


# -------------------------------------------------------------------
# 5. Week 4 entrypoint (called from main.py / week6 pipeline)
# -------------------------------------------------------------------
def run_week4_hqnn():
    datasets = load_drm_datasets()
    hybrid_results = run_hybrid_qnn_experiments(datasets)
    classical_results, comparison_df = run_classical_models(
        datasets, hybrid_results
    )
    return hybrid_results, classical_results, comparison_df


if __name__ == "__main__":
    run_week4_hqnn()
