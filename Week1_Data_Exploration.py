import os
import tarfile
import urllib.request
from fairchem.core.datasets.lmdb_dataset import LmdbDataset

# === Local storage path ===
data_dir = r"E:\Computational Engineering\samples"
os.makedirs(data_dir, exist_ok=True)

url = "https://dl.fbaipublicfiles.com/opencatalystproject/data/is2res_train_val_test_lmdbs.tar.gz"
tar_path = os.path.join(data_dir, "is2res_train_val_test_lmdbs.tar.gz")

# === Step 1: Streamed + resumable download ===
def download_file(url, dest, chunk_size=16 * 1024 * 1024):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    mode = "ab" if os.path.exists(dest) else "wb"
    downloaded = os.path.getsize(dest) if os.path.exists(dest) else 0
    print(f"⬇️ Starting download (resume from {downloaded / (1024 * 1024):.2f} MB)...")

    with urllib.request.urlopen(req) as response, open(dest, mode) as out_file:
        total_size = response.length or 25 * 1024 * 1024 * 1024  # Approx 25 GB if unknown
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            pct = (downloaded / total_size) * 100
            print(f"📦 {downloaded / (1024 * 1024):.2f} MB ({pct:.2f}%)", end="\r")
    print("\n✅ Download complete.")

if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 1024 * 1024:
    download_file(url, tar_path)
else:
    print(f"✅ File already exists ({os.path.getsize(tar_path) / (1024 * 1024):.2f} MB).")

# === Step 2: Verify before extraction ===
if os.path.getsize(tar_path) < 1024 * 1024:
    raise ValueError("❌ Download incomplete — file too small!")

extracted_dir = os.path.join(data_dir, "is2res_lmdbs")

if not os.path.exists(extracted_dir):
    print("📦 Extracting dataset...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extracted_dir)
        print("✅ Extraction complete.")
    except tarfile.ReadError:
        raise ValueError("❌ Extraction failed — file may be incomplete or corrupted.")
else:
    print("✅ Dataset already extracted.")

# === Step 3: Locate LMDB directories ===
train_dir = None
for root, dirs, files in os.walk(extracted_dir):
    if any(f.endswith(".lmdb") for f in files):
        train_dir = root
        break

if not train_dir:
    raise FileNotFoundError("❌ No LMDB files found after extraction.")
print(f"✅ Found LMDB path: {train_dir}")

# === Step 4: Load dataset ===
config = {"src": train_dir, "train": True}
ds = LmdbDataset(config)

print(f"✅ Dataset loaded successfully — {len(ds)} samples found")
print("🧪 Example keys:", list(ds[0].keys()))
