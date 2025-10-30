from fairchem.core.datasets import AseDBDataset
import os

# ✅ Local data directory
data_dir = r"E:\Computational Engineering\samples"
db_path = os.path.join(data_dir, "oc20_sample.db")

# ✅ Create a configuration dictionary
config = {
    "src": db_path,   # source path for ASE database
    "train": False,   # not in training mode
}

# ✅ Initialize the dataset with the config dictionary
ds = AseDBDataset(config)

print("✅ Dataset initialized successfully!")
print(f"Number of data points: {len(ds)}")

# ✅ Access the first sample
sample = ds[0]
print(f"Atoms in sample: {len(sample.atoms)}")
print("Available keys:", list(sample.keys()))
