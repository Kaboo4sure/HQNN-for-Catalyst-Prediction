from fairchem.core.datasets import AseDBDataset
import os

# ✅ Local data directory (you can change this path)
data_dir = r"E:\Computational Engineering\samples"

# Example ASE database file (dummy or real)
db_path = os.path.join(data_dir, "oc20_sample.db")

# Initialize dataset — FairChem v1.10.0 expects db_paths only
ds = AseDBDataset(db_paths=[db_path])

print("✅ Dataset initialized successfully!")
print(f"Number of data points: {len(ds)}")

# Access the first record
sample = ds[0]
print(f"Atoms in sample: {len(sample.atoms)}")
print("Available keys:", list(sample.keys()))
