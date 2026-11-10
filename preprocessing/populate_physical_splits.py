"""
Safely populate physical dataset/train and dataset/test directories for college demonstration.
Creates:
  dataset/train/image/{real, fake}
  dataset/train/video/{real, fake}
  dataset/train/audio/{real, fake}
  dataset/test/image/{real, fake}
  dataset/test/video/{real, fake}
  dataset/test/audio/{real, fake}

And mirrors to:
  dataset/{image,video,audio}/train/{real, fake}
  dataset/{image,video,audio}/test/{real, fake}
so both folder conventions work seamlessly.
"""

import os
import shutil
import pandas as pd
from pathlib import Path

WORKSPACE = Path(r"D:\deepfake_college_project\Deepfake-Analysis")
MANIFEST_PATH = WORKSPACE / "reports" / "dataset_manifest.csv"
DATASET_DIR = WORKSPACE / "dataset"

def populate_splits():
    print("Loading manifest:", MANIFEST_PATH)
    df = pd.read_csv(MANIFEST_PATH)
    valid_df = df[df["is_valid"] == True]
    
    # Target counts per class for a balanced, visually clean presentation
    limits = {
        ("image", "train", "Real"): 700,
        ("image", "train", "Fake"): 700,
        ("image", "test", "Real"): 150,
        ("image", "test", "Fake"): 150,

        ("video", "train", "Real"): 100,
        ("video", "train", "Fake"): 100,
        ("video", "test", "Real"): 30,
        ("video", "test", "Fake"): 30,

        ("audio", "train", "Real"): 250,
        ("audio", "train", "Fake"): 250,
        ("audio", "test", "Real"): 50,
        ("audio", "test", "Fake"): 50,
    }

    copied_stats = {}

    for (modality, split, label), max_count in limits.items():
        subset = valid_df[
            (valid_df["modality"] == modality) & 
            (valid_df["split"] == split) & 
            (valid_df["label"] == label)
        ]
        
        sample_paths = subset["filepath"].head(max_count).tolist()
        
        # Structure 1: dataset/{split}/{modality}/{label_lower}
        dest_dir1 = DATASET_DIR / split / modality / label.lower()
        dest_dir1.mkdir(parents=True, exist_ok=True)

        # Structure 2: dataset/{modality}/{split}/{label_lower}
        dest_dir2 = DATASET_DIR / modality / split / label.lower()
        dest_dir2.mkdir(parents=True, exist_ok=True)

        count = 0
        total_bytes = 0
        for src in sample_paths:
            src_p = Path(src)
            if not src_p.exists():
                continue
            
            # File name collision safety: prefix with index if needed
            fname = src_p.name
            target1 = dest_dir1 / fname
            target2 = dest_dir2 / fname

            if not target1.exists():
                shutil.copy2(src_p, target1)
                total_bytes += src_p.stat().st_size
            
            if not target2.exists():
                shutil.copy2(src_p, target2)
                total_bytes += src_p.stat().st_size

            count += 1

        copied_stats[(split, modality, label)] = (count, total_bytes)
        print(f"Copied {count} files for {split}/{modality}/{label} ({total_bytes / (1024**2):.1f} MB)")

    print("\n================ POPULATION SUMMARY ================")
    total_files = sum(c for c, _ in copied_stats.values()) // 2
    total_sz = sum(b for _, b in copied_stats.values())
    print(f"Total Unique Files Populated: {total_files}")
    print(f"Total Physical Storage Used: {total_sz / (1024**2):.1f} MB ({total_sz / (1024**3):.2f} GB)")
    print("====================================================")

if __name__ == "__main__":
    populate_splits()
