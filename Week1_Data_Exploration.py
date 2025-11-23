import os
import tarfile
import urllib.request
import torch
import pandas as pd
import json
from fairchem.core.datasets.lmdb_dataset import LmdbDataset

# === Base storage directory ===
data_dir = r"E:\Computational Engineering\samples"
os.makedirs(data_dir, exist_ok=True)

# === New dataset (tutorial_data) ===
url = "http://dl.fbaipublicfiles.com/opencatalystproject/data/tutorial_data.tar.gz"
tar_path = os.path.join(data_dir, "tutorial_data.tar.gz")
extract_dir = os.path.join(data_dir, "tutorial_data_extracted")
csv_dir = os.path.join(data_dir, "tutorial_csv")
os.makedirs(csv_dir, exist_ok=True)

# =================================================================
# Step 1 — Download New File (Supports Resume)
# =================================================================
def download_file(url, dest, chunk_size=16 * 1024 * 1024):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    mode = "ab" if os.path.exists(dest) else "wb"
    downloaded = os.path.getsize(dest) if os.path.exists(dest) else 0
    print(f"⬇️ Starting download (resume from {downloaded/(1024*1024):.2f} MB)...")

    with urllib.request.urlopen(req) as response, open(dest, mode) as out_file:
        total_size = response.length or 2 * 1024 * 1024 * 1024  # Unknown size fallback
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            pct = (downloaded / total_size) * 100
            print(f"📦 {downloaded/(1024*1024):.2f} MB ({pct:.2f}%)", end="\r")

    print("\n✅ Download complete.")


# === Download only if missing or small ===
if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1024 * 1024:
    download_file(url, tar_path)
else:
    print(f"✅ tutorial_data.tar.gz already exists ({os.path.getsize(tar_path)/(1024*1024):.2f} MB).")


# =================================================================
# Step 2 — Extract to NEW folder
# =================================================================
if not os.path.exists(extract_dir):
    print("📦 Extracting new tutorial_data...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        print("✅ Extraction complete:", extract_dir)
    except tarfile.ReadError:
        raise ValueError("❌ Extraction failed — file may be corrupted.")
else:
    print("✅ tutorial_data already extracted!")

# =================================================================
# Step 3 — Search manually for LMDB files
# =================================================================
lmdb_paths = []

for root, dirs, files in os.walk(extract_dir):
    if "data.lmdb" in files:
        lmdb_paths.append(root)

print("\n🔍 LMDB datasets found:")
for p in lmdb_paths:
    print("  •", p)

# =================================================================
# Step 4 — Convert LMDB → CSV and Merge
# =================================================================
all_rows = []

def read_lmdb(lmdb_path, label_name, max_samples=300):
    print(f"\n📖 Reading LMDB: {lmdb_path}")
    ds = LmdbDataset({"src": lmdb_path, "train": False})
    total = len(ds)
    print(f"📦 Total samples: {total}")

    rows = []

    for i in range(min(total, max_samples)):
        sample = ds[i]
        row = {"split": label_name, "index": i}

        for k, v in sample.items():
            try:
                row[k] = v.tolist() if hasattr(v, "tolist") else v
            except:
                row[k] = str(v)

        rows.append(row)

    return rows


# Read each LMDB and merge
for path in lmdb_paths:
    label = path.replace(extract_dir + "\\", "").replace("\\", "_")
    rows = read_lmdb(path, label)
    all_rows.extend(rows)


# Convert to BIG CSV
merged_csv = os.path.join(csv_dir, "tutorial_data_merged.csv")
df = pd.DataFrame(all_rows)
df.to_csv(merged_csv, index=False)

print("\n🎉 MERGED CSV CREATED:", merged_csv)
print(f"📊 Total rows: {len(df)}")
