import os
import requests
import tarfile
from fairchem.core.datasets.lmdb_dataset import LmdbDataset

data_dir = r"E:\Computational Engineering\samples"
os.makedirs(data_dir, exist_ok=True)

# HF mirror of OC20 sample
url = "https://huggingface.co/datasets/FAIR/Open-Catalyst-Project/resolve/main/2020-05-18-oc20_sample.tar.gz"
tar_path = os.path.join(data_dir, "oc20_sample.tar.gz")

if not os.path.exists(os.path.join(data_dir, "oc20_sample")):
    print("⬇️ Downloading OC20 sample dataset from Hugging Face...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(tar_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("✅ Download complete. Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)
    print("✅ Extraction finished.")
else:
    print("✅ Dataset already present.")

config = {"src": os.path.join(data_dir, "oc20_sample"), "train": False}
ds = LmdbDataset(config)
print(f"✅ Dataset loaded successfully — {len(ds)} samples")
print("Keys:", list(ds[0].keys()))
