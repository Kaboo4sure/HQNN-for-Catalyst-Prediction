from fairchem.core.datasets import AseDBDataset
from pathlib import Path

# define a small local dataset directory
data_dir = Path("E:\Computational Engineering/samples")
data_dir.mkdir(parents=True, exist_ok=True)

# test load of a sample dataset (this one ships with fairchem)
ds = AseDBDataset("OC20_sample", root=data_dir)

print(f"Dataset loaded at: {data_dir}")
print(f"Number of samples: {len(ds)}")
print(f"Sample keys: {list(ds[0].keys())}")
