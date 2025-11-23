import os
import tarfile
import urllib.request
import torch
import pandas as pd
import json

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
    print(f"⬇️ Starting download (resume from {downloaded / (1024*1024):.2f} MB)...")

    with urllib.request.urlopen(req) as response, open(dest, mode) as out_file:
        total_size = response.length or 2 * 1024 * 1024 * 1024  # 2GB estimate
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            pct = (downloaded / total_size) * 100
            print(f"📦 {downloaded / (1024*1024):.2f} MB ({pct:.2f}%)", end="\r")

    print("\n✅ Download complete.")


# === Only download if missing or too small ===
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
# Step 3 — Search for .pt or .json files
# =================================================================
pt_files = []
json_files = []

for root, dirs, files in os.walk(extract_dir):
    for f in files:
        if f.endswith(".pt"):
            pt_files.append(os.path.join(root, f))
        if f.endswith(".json"):
            json_files.append(os.path.join(root, f))

print("🔍 Found PT files:", len(pt_files))
print("🔍 Found JSON files:", len(json_files))


# =================================================================
# Step 4A — Convert .pt files → CSV
# =================================================================
def convert_pt_to_csv(pt_file, out_csv):
    print(f"📄 Converting {pt_file} → CSV...")
    data = torch.load(pt_file)

    flat = {}
    for k, v in data.items():
        try:
            flat[k] = v.numpy().flatten()
        except Exception:
            flat[k] = [v]

    df = pd.DataFrame(flat)
    df.to_csv(out_csv, index=False)
    print("✅ Saved:", out_csv)


for pt in pt_files:
    out_csv = os.path.join(csv_dir, os.path.basename(pt).replace(".pt", ".csv"))
    convert_pt_to_csv(pt, out_csv)


# =================================================================
# Step 4B — Convert .json files → CSV
# =================================================================
def convert_json_to_csv(json_file, out_csv):
    print(f"📄 Converting {json_file} → CSV...")
    with open(json_file, "r") as f:
        data = json.load(f)

    df = pd.json_normalize(data)
    df.to_csv(out_csv, index=False)
    print("✅ Saved:", out_csv)


for js in json_files:
    out_csv = os.path.join(csv_dir, os.path.basename(js).replace(".json", ".csv"))
    convert_json_to_csv(js, out_csv)


print("\n🎉 ALL DONE — CSV files saved into:", csv_dir)
