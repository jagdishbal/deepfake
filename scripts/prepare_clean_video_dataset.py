"""
Extract clean, verified Video samples for Deepfake Analysis.
Guarantees:
1. Ground truth correctness:
   - Real Video: Sourced from RealVideo-RealAudio and RealVideo-FakeAudio (video track is authentic)
   - Fake Video: Sourced from FakeVideo-RealAudio (FaceSwap/FSGAN) and FakeVideo-FakeAudio (Wav2Lip)
2. Zero speaker leakage:
   - Stratified partition by speaker identity (source ID)
   - Train speakers (350), Val speakers (75), Test speakers (75) are mutually exclusive.
3. Clean MP4 videos copied to dataset/video.
"""

import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path

WORKSPACE = Path(r"D:\deepfake_college_project\Deepfake-Analysis")
RAW_ROOT = WORKSPACE / "dataset" / "raw" / "videos" / "audio_dataset.zio" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"
META_CSV = WORKSPACE / "dataset" / "raw" / "audio" / "audio_dataset.zio" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2" / "meta_data.csv"
OUTPUT_DIR = WORKSPACE / "dataset" / "video"


def main():
    print("=" * 60)
    print("PREPARING CLEAN VERIFIED VIDEO DATASET")
    print("=" * 60)

    # 1. Read metadata and group by speaker ID
    records_by_speaker = defaultdict(lambda: {"real": [], "fake": []})
    with open(META_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            typ = row["type"]
            race = row["race"]
            gender = row["gender"]
            src = row["source"]
            fname = row["path"]
            mp4_path = RAW_ROOT / typ / race / gender / src / fname
            if not mp4_path.exists():
                continue

            if typ in ["RealVideo-RealAudio", "RealVideo-FakeAudio"]:
                records_by_speaker[src]["real"].append(mp4_path)
            elif typ in ["FakeVideo-RealAudio", "FakeVideo-FakeAudio"]:
                records_by_speaker[src]["fake"].append(mp4_path)

    speakers = sorted(records_by_speaker.keys())
    print(f"Total valid speakers: {len(speakers)}")

    rng = random.Random(42)
    rng.shuffle(speakers)

    train_speakers = set(speakers[:350])
    val_speakers = set(speakers[350:425])
    test_speakers = set(speakers[425:])

    print(f"Speakers split: Train={len(train_speakers)}, Val={len(val_speakers)}, Test={len(test_speakers)}")

    split_targets = {
        "train": (train_speakers, 200, 200),
        "validation": (val_speakers, 40, 40),
        "test": (test_speakers, 40, 40),
    }

    # Clean destination folders
    for split in ["train", "validation", "test"]:
        for label in ["real", "fake"]:
            p = OUTPUT_DIR / split / label
            p.mkdir(parents=True, exist_ok=True)
            for existing in p.glob("*"):
                if existing.is_file():
                    existing.unlink()

    copied_total = 0
    for split, (split_spkrs, target_real, target_fake) in split_targets.items():
        pool_real = []
        pool_fake = []
        for spkr in sorted(split_spkrs):
            pool_real.extend(records_by_speaker[spkr]["real"])
            pool_fake.extend(records_by_speaker[spkr]["fake"])

        rng.shuffle(pool_real)
        rng.shuffle(pool_fake)

        selected_real = pool_real[:target_real]
        selected_fake = pool_fake[:target_fake]

        print(f"{split.upper()}: Copying {len(selected_real)} Real, {len(selected_fake)} Fake videos...")

        for idx, src_p in enumerate(selected_real):
            dst = OUTPUT_DIR / split / "real" / f"{src_p.stem}_real_{idx:03d}.mp4"
            shutil.copy2(src_p, dst)
            copied_total += 1

        for idx, src_p in enumerate(selected_fake):
            dst = OUTPUT_DIR / split / "fake" / f"{src_p.stem}_fake_{idx:03d}.mp4"
            shutil.copy2(src_p, dst)
            copied_total += 1

    print(f"\nCopied {copied_total} clean video files successfully!")

    print("\nDataset Summary:")
    for split in ["train", "validation", "test"]:
        for label in ["real", "fake"]:
            p = OUTPUT_DIR / split / label
            count = sum(1 for _ in p.glob("*.mp4"))
            print(f"  {split}/{label}: {count} MP4 files")


if __name__ == "__main__":
    main()
