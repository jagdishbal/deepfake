import os
import sys
import json
import urllib.request
import io
from pathlib import Path
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
STAGING_DIR = PROJECT_ROOT / "dataset" / "benchmark_cifake"

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def download_file(url, dest_path):
    print(f"Downloading from {url} to {dest_path}...")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest_path, "wb") as f:
        total = 0
        while True:
            chunk = resp.read(1024 * 64)
            if not chunk:
                break
            f.write(chunk)
            total += len(chunk)
            if total % (1024 * 1024 * 5) == 0:
                print(f"  Downloaded {total / (1024*1024):.1f} MB...")
    print(f"Finished downloading {dest_path.name} ({total / (1024*1024):.2f} MB)")

def get_parquet_urls():
    candidates = [
        "https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
        "https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset/resolve/main/data/train-00000-of-00001.parquet",
    ]
    # First query API
    api_url = "https://huggingface.co/api/datasets/dragonintelligence/CIFAKE-image-dataset"
    try:
        req = urllib.request.Request(api_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=15) as r:
            info = json.loads(r.read().decode("utf-8"))
            for s in info.get("siblings", []):
                rfile = s.get("rfilename", "")
                if rfile.endswith(".parquet"):
                    url = f"https://huggingface.co/datasets/dragonintelligence/CIFAKE-image-dataset/resolve/main/{rfile}"
                    if url not in candidates:
                        candidates.insert(0, url)
    except Exception as e:
        print(f"API query note: {e}")

    return candidates

def extract_and_save_samples(parquet_path, max_per_class=600):
    print(f"Reading parquet file: {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Parquet loaded! Total rows: {len(df)}, columns: {df.columns.tolist()}")

    # Determine columns
    img_col = "image" if "image" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else df.columns[1]

    print(f"Using img_col='{img_col}', label_col='{label_col}'")
    print(f"Label value counts:\n{df[label_col].value_counts()}")

    # In CIFAKE: 0 = FAKE (Stable Diffusion), 1 = REAL (CIFAR-10)
    fake_df = df[df[label_col] == 0]
    real_df = df[df[label_col] == 1]

    # Target splits: 70% train, 15% validation, 15% test
    n_train = int(max_per_class * 0.70)
    n_val = int(max_per_class * 0.15)
    n_test = max_per_class - n_train - n_val

    splits = {
        "train": (0, n_train),
        "validation": (n_train, n_train + n_val),
        "test": (n_train + n_val, max_per_class)
    }

    for split_name, (start_idx, end_idx) in splits.items():
        fake_split = fake_df.iloc[start_idx:end_idx]
        real_split = real_df.iloc[start_idx:end_idx]

        for cls_name, sub_df in [("fake", fake_split), ("real", real_split)]:
            target_dir = STAGING_DIR / split_name / cls_name
            target_dir.mkdir(parents=True, exist_ok=True)

            saved = 0
            for idx, row in sub_df.iterrows():
                img_data = row[img_col]
                img_bytes = img_data.get("bytes") if isinstance(img_data, dict) else img_data
                if not isinstance(img_bytes, (bytes, bytearray)):
                    continue
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                    # Save as PNG
                    save_path = target_dir / f"cifake_{cls_name}_{idx:06d}.png"
                    img.save(save_path, "PNG")
                    saved += 1
                except Exception as exc:
                    pass
            print(f"Saved {saved} {cls_name} images into {target_dir}")

def main():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    parquet_path = STAGING_DIR / "cifake_train.parquet"

    if not parquet_path.exists() or parquet_path.stat().st_size == 0:
        urls = get_parquet_urls()
        success = False
        for url in urls:
            try:
                download_file(url, parquet_path)
                if parquet_path.stat().st_size > 1024 * 1024:
                    success = True
                    break
            except Exception as e:
                print(f"Failed URL {url}: {e}")
        if not success:
            print("Error: Could not download parquet file.")
            sys.exit(1)

    extract_and_save_samples(parquet_path, max_per_class=600)
    print("CIFAKE benchmark extraction complete!")

if __name__ == "__main__":
    main()
