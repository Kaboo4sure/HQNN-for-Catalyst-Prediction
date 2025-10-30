from fairchem.core.datasets import AseDBDataset
import os

# ✅ Local data directory (adjust if needed)
data_dir = r"E:\Computational Engineering\samples"
db_path = os.path.join(data_dir, "oc20_sample.db")

# ✅ Initialize the dataset (v1.10.0 expects a single path, not db_paths)
ds = AseDBDataset(db_path)

print("✅ Dataset initialized successfully!")
print(f"Number of data points: {len(ds)}")

# ✅ Inspect the first sample
sample = ds[0]
print(f"Atoms in sample: {len(sample.atoms)}")
print("Available keys:", list(sample.keys()))
