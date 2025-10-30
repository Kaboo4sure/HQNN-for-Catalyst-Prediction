from fairchem.core.datasets.lmdb_dataset import LmdbDataset
from ase import Atoms
import os

# ✅ Local path to your dummy dataset
data_dir = r"E:\Computational Engineering\samples"
db_path = os.path.join(data_dir, "oc20_sample.db")

# ✅ Config structure for FairChem v1.10.0
config = {
    "src": db_path,
    "train": False,
}

# ✅ Initialize the LMDB-style dataset
ds = LmdbDataset(config)

print("✅ Dataset initialized successfully!")
print(f"Number of data points: {len(ds)}")

# ✅ Retrieve a sample
sample = ds[0]
print(f"Available keys: {list(sample.keys())}")
print(f"Energy: {sample.get('energy', 'N/A')}")
print(f"Forces: {sample.get('forces', 'N/A')}")
