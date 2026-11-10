"""
Extract clean, verified Audio samples for Deepfake Analysis.
Uses strictly:
- Real Audio: RealVideo-RealAudio (500 authentic celebrity speech recordings)
- Fake Audio: RealVideo-FakeAudio (500 RTVC synthetic voice-cloned speech recordings)
With zero speaker leakage across train (350), validation (75), and test (75).
"""

import csv
import random
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from imageio_ffmpeg import get_ffmpeg_exe

WORKSPACE = Path(r"D:\deepfake_college_project\Deepfake-Analysis")
RAW_ROOT = WORKSPACE / "dataset" / "raw" / "videos" / "audio_dataset.zio" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"
META_CSV = WORKSPACE / "dataset" / "raw" / "audio" / "audio_dataset.zio" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2" / "meta_data.csv"
OUTPUT_DIR = WORKSPACE / "dataset" / "audio"

TARGET_SR = 16000
DURATION = 3.0


def extract_clip(src_mp4: Path, dst_wav: Path, ffmpeg_exe: str) -> bool:
    """Extract 3-second mono 16kHz PCM WAV using FFmpeg."""
    dst_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-y",
        "-ss", "0.5",
        "-t", str(DURATION),
        "-i", str(src_mp4),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(TARGET_SR),
        "-ac", "1",
        str(dst_wav),
    ]
    res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if res.returncode != 0 or not dst_wav.exists() or dst_wav.stat().st_size == 0:
        cmd[3] = "0.0"
        res = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True)
    return dst_wav.exists() and dst_wav.stat().st_size > 500


def main():
    print("=" * 60)
    print("PREPARING STRICTLY PAIRED REAL & RTVC FAKE AUDIO DATASET")
    print("=" * 60)

    ffmpeg_exe = get_ffmpeg_exe()

    # Find paired Real and RTVC fake for all 500 speakers
    pairs = {}
    with open(META_CSV, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row["source"]
            typ = row["type"]
            if typ in ["RealVideo-RealAudio", "RealVideo-FakeAudio"]:
                p = RAW_ROOT / typ / row["race"] / row["gender"] / src / row["path"]
                if p.exists():
                    if src not in pairs:
                        pairs[src] = {}
                    pairs[src][typ] = p

    complete_speakers = sorted([s for s in pairs if len(pairs[s]) == 2])
    print(f"Total speakers with verified Real & Fake audio pairs: {len(complete_speakers)}")

    # Deterministic split: 350 Train, 75 Validation, 75 Test
    rng = random.Random(42)
    rng.shuffle(complete_speakers)

    train_spkrs = complete_speakers[:350]
    val_spkrs = complete_speakers[350:425]
    test_spkrs = complete_speakers[425:500]

    splits = {
        "train": train_spkrs,
        "validation": val_spkrs,
        "test": test_spkrs,
    }

    # Clean folders
    for split in ["train", "validation", "test"]:
        for label in ["real", "fake"]:
            p = OUTPUT_DIR / split / label
            p.mkdir(parents=True, exist_ok=True)
            for f in p.glob("*"):
                if f.is_file():
                    f.unlink()

    tasks = []
    for split, spkrs in splits.items():
        for spkr in spkrs:
            real_p = pairs[spkr]["RealVideo-RealAudio"]
            fake_p = pairs[spkr]["RealVideo-FakeAudio"]

            dst_real = OUTPUT_DIR / split / "real" / f"{spkr}_{real_p.stem}.wav"
            dst_fake = OUTPUT_DIR / split / "fake" / f"{spkr}_{fake_p.stem}.wav"

            tasks.append((real_p, dst_real, ffmpeg_exe))
            tasks.append((fake_p, dst_fake, ffmpeg_exe))

    print(f"Extracting {len(tasks)} audio clips ({len(train_spkrs)*2} train, {len(val_spkrs)*2} val, {len(test_spkrs)*2} test)...")
    completed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(extract_clip, s, d, f) for s, d, f in tasks]
        for fut in futures:
            if fut.result():
                completed += 1

    print(f"Extracted {completed}/{len(tasks)} audio files successfully!")

    print("\nDataset Summary:")
    for split in ["train", "validation", "test"]:
        for label in ["real", "fake"]:
            p = OUTPUT_DIR / split / label
            count = sum(1 for _ in p.glob("*.wav"))
            print(f"  {split}/{label}: {count} WAV files")


if __name__ == "__main__":
    main()
