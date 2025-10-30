from fairchem.core.datasets import AseDBDataset
import os

# ✅ Local data directory (you can change this path)
data_dir = r"E:\Computational Engineering/samples"

# Example LMDB or ASE database file (you can later replace with real OC20 or custom data)
db_path = os.path.join(data_dir, "oc20_sample.db")

# Initialize dataset properly for FairChem v1.10.0
ds = AseDBDataset(name="OC20_sample", db_paths=[db_path])

print("✅ Dataset initialized successfully!")
print(f"Number of data points: {len(ds)}")

# Show sample structure info
sample = ds[0]
print(f"Atoms in sample: {len(sample.atoms)}")
print("Available keys:", sample.keys())
