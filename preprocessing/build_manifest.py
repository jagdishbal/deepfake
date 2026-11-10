"""
Build Dataset Manifest & Non-Destructive Integrity Cleaning
Generates:
  - reports/dataset_manifest.csv [filepath, modality, label, is_valid, split]
  - reports/corrupted_files.txt
"""

import os
import sys
import csv
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split

WORKSPACE_ROOT = Path(r"D:\deepfake_college_project\Deepfake-Analysis")
RAW_ROOT = WORKSPACE_ROOT / "dataset" / "raw"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
METADATA_CSV = RAW_ROOT / "audio" / "audio_dataset.zio" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2" / "meta_data.csv"

def verify_image(filepath_str: str) -> tuple[bool, str]:
    try:
        with Image.open(filepath_str) as img:
            img.verify()
        return True, ""
    except Exception as e:
        return False, f"PIL verify failed: {e}"

def verify_video(filepath_str: str) -> tuple[bool, str]:
    try:
        cap = cv2.VideoCapture(filepath_str)
        if not cap.isOpened():
            cap.release()
            return False, "cv2 failed to open video"
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if frames <= 0:
            cap.release()
            return False, f"Invalid frame count: {frames}"
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return False, "Failed to read first frame"
        return True, ""
    except Exception as e:
        return False, f"cv2 exception: {e}"

def verify_audio_mp4(filepath_str: str) -> tuple[bool, str]:
    try:
        with open(filepath_str, 'rb') as f:
            header = f.read(1024 * 1024) # first 1MB
            if b'soun' in header:
                return True, ""
            return False, "No audio atom ('soun') found in MP4 header"
    except Exception as e:
        return False, f"File read error: {e}"

def check_batch_images(paths):
    results = []
    for p in paths:
        valid, err = verify_image(p)
        results.append((p, valid, err))
    return results

def check_batch_videos(paths):
    results = []
    for p in paths:
        valid, err = verify_video(p)
        results.append((p, valid, err))
    return results

def check_batch_audio(paths):
    results = []
    for p in paths:
        valid, err = verify_audio_mp4(p)
        results.append((p, valid, err))
    return results

def main():
    start_time = time.time()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PHASE 2 & 3: DATASET INVENTORY & INTEGRITY CLEANING")
    print("=" * 60)

    # Load FakeAVCeleb metadata
    print("Loading FakeAVCeleb metadata...")
    if not METADATA_CSV.exists():
        print(f"Error: metadata.csv not found at {METADATA_CSV}")
        return

    meta_df = pd.read_csv(METADATA_CSV)
    # Build lookup by video filename or relative path
    # 'path' in meta_data is e.g.: FakeAVCeleb/RealVideo-RealAudio/African/men/id00076/00109.mp4
    # The actual file on disk is e.g. 00109_10_id00476_wavtolip.mp4 or similar
    # Let's inspect how filename maps to metadata row
    print(f"Loaded metadata entries: {len(meta_df)}")
    
    raw_images_list = [] # (path, label)
    raw_videos_list = [] # (path, label)
    raw_audio_list = []  # (path, label)

    # 1. IMAGES
    print("Indexing images...")
    # FakeAVCeleb frames
    frames_dir = RAW_ROOT / "images" / "audio_dataset.zio" / "FakeAVCeleb_v1.2" / "frames"
    if frames_dir.exists():
        for category_dir in frames_dir.iterdir():
            if not category_dir.is_dir():
                continue
            cat_name = category_dir.name
            if "RealVideo" in cat_name:
                img_label = "Real"
            else:
                img_label = "Fake"
            for root, _, files in os.walk(category_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                        raw_images_list.append((os.path.join(root, f), img_label))

    # FaceForensics Deepfakes frames
    deepfakes_dir = RAW_ROOT / "images" / "video_dataset" / "fake"
    if deepfakes_dir.exists():
        for root, _, files in os.walk(deepfakes_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.png', '.jpeg', '.webp')):
                    raw_images_list.append((os.path.join(root, f), "Fake"))

    print(f"Found {len(raw_images_list)} images total.")

    # 2. VIDEOS & AUDIO
    print("Indexing videos and audio...")
    # DFD Videos
    dfd_real = RAW_ROOT / "videos" / "image_dataset" / "DFD_original sequences"
    if dfd_real.exists():
        for root, _, files in os.walk(dfd_real):
            for f in files:
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    raw_videos_list.append((os.path.join(root, f), "Real"))

    dfd_fake = RAW_ROOT / "videos" / "image_dataset" / "DFD_manipulated_sequences"
    if dfd_fake.exists():
        for root, _, files in os.walk(dfd_fake):
            for f in files:
                if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    raw_videos_list.append((os.path.join(root, f), "Fake"))

    # FakeAVCeleb Videos and Audio
    fakeav_vid_dir = RAW_ROOT / "videos" / "audio_dataset.zio" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2"
    if fakeav_vid_dir.exists():
        for root, _, files in os.walk(fakeav_vid_dir):
            for f in files:
                if f.lower().endswith('.mp4'):
                    full_p = os.path.join(root, f)
                    # Check folder category in path
                    norm_p = full_p.replace("\\", "/")
                    if "/RealVideo-RealAudio/" in norm_p:
                        v_label = "Real"
                        a_label = "Real"
                    elif "/RealVideo-FakeAudio/" in norm_p:
                        v_label = "Real"
                        a_label = "Fake"
                    elif "/FakeVideo-RealAudio/" in norm_p:
                        v_label = "Fake"
                        a_label = "Real"
                    elif "/FakeVideo-FakeAudio/" in norm_p:
                        v_label = "Fake"
                        a_label = "Fake"
                    else:
                        # Fallback inference
                        v_label = "Fake"
                        a_label = "Fake"

                    raw_videos_list.append((full_p, v_label))
                    raw_audio_list.append((full_p, a_label))

    print(f"Found {len(raw_videos_list)} videos total.")
    print(f"Found {len(raw_audio_list)} audio sources total.")

    print("\n--- Integrity Verification Pass ---")
    workers = min(os.cpu_count() or 4, 10)
    batch_size = 500

    corrupted_entries = []
    validated_records = [] # (path, modality, label, is_valid)

    # Validate Images
    print(f"Validating {len(raw_images_list)} images using {workers} worker processes...")
    img_batches = [
        [p for p, _ in raw_images_list[i:i + batch_size]]
        for i in range(0, len(raw_images_list), batch_size)
    ]
    img_label_map = {p: lbl for p, lbl in raw_images_list}

    completed_imgs = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(check_batch_images, b): b for b in img_batches}
        for future in as_completed(futures):
            batch_res = future.result()
            for p, valid, err in batch_res:
                lbl = img_label_map[p]
                validated_records.append((p, "image", lbl, valid))
                if not valid:
                    corrupted_entries.append((p, "image", err))
            completed_imgs += len(batch_res)
            if completed_imgs % 20000 == 0 or completed_imgs == len(raw_images_list):
                print(f"  Images verified: {completed_imgs}/{len(raw_images_list)}")

    # Validate Videos
    print(f"Validating {len(raw_videos_list)} videos using {workers} worker processes...")
    vid_batches = [
        [p for p, _ in raw_videos_list[i:i + batch_size]]
        for i in range(0, len(raw_videos_list), batch_size)
    ]
    vid_label_map = {p: lbl for p, lbl in raw_videos_list}

    completed_vids = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(check_batch_videos, b): b for b in vid_batches}
        for future in as_completed(futures):
            batch_res = future.result()
            for p, valid, err in batch_res:
                lbl = vid_label_map[p]
                validated_records.append((p, "video", lbl, valid))
                if not valid:
                    corrupted_entries.append((p, "video", err))
            completed_vids += len(batch_res)
            if completed_vids % 5000 == 0 or completed_vids == len(raw_videos_list):
                print(f"  Videos verified: {completed_vids}/{len(raw_videos_list)}")

    # Validate Audio
    print(f"Validating {len(raw_audio_list)} audio tracks using {workers} worker processes...")
    aud_batches = [
        [p for p, _ in raw_audio_list[i:i + batch_size]]
        for i in range(0, len(raw_audio_list), batch_size)
    ]
    aud_label_map = {p: lbl for p, lbl in raw_audio_list}

    completed_aud = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(check_batch_audio, b): b for b in aud_batches}
        for future in as_completed(futures):
            batch_res = future.result()
            for p, valid, err in batch_res:
                lbl = aud_label_map[p]
                validated_records.append((p, "audio", lbl, valid))
                if not valid:
                    corrupted_entries.append((p, "audio", err))
            completed_aud += len(batch_res)
            if completed_aud % 5000 == 0 or completed_aud == len(raw_audio_list):
                print(f"  Audio verified: {completed_aud}/{len(raw_audio_list)}")

    print(f"\nTotal corrupted/unreadable files: {len(corrupted_entries)}")

    # Write corrupted files report
    corrupted_txt_path = REPORTS_DIR / "corrupted_files.txt"
    with open(corrupted_txt_path, "w", encoding="utf-8") as f:
        f.write(f"# CORRUPTED FILES REPORT - Generated {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"# Total Corrupted / Unreadable: {len(corrupted_entries)}\n\n")
        for p, modality, err in corrupted_entries:
            f.write(f"[{modality.upper()}] {p} -> {err}\n")
    print(f"Corrupted files log saved to {corrupted_txt_path}")

    # Assign Stratified Splits (70% train, 15% validation, 15% test)
    print("\n--- Assigning Stratified Splits (Train: 70%, Val: 15%, Test: 15%) ---")
    manifest_rows = []
    
    # Group by modality
    by_modality = defaultdict(list)
    for p, modality, lbl, is_valid in validated_records:
        by_modality[modality].append((p, lbl, is_valid))

    for modality, items in by_modality.items():
        valid_items = [it for it in items if it[2]] # is_valid == True
        invalid_items = [it for it in items if not it[2]] # is_valid == False

        # Add invalid items with split='none'
        for p, lbl, _ in invalid_items:
            manifest_rows.append({
                "filepath": p,
                "modality": modality,
                "label": lbl,
                "is_valid": False,
                "split": "none"
            })

        # Stratified train/val/test split on valid items
        paths = [it[0] for it in valid_items]
        labels = [it[1] for it in valid_items]

        # 70% train, 30% temp (which will be 15% val, 15% test)
        try:
            train_p, temp_p, train_l, temp_l = train_test_split(
                paths, labels, test_size=0.30, random_state=42, stratify=labels
            )
            val_p, test_p, val_l, test_l = train_test_split(
                temp_p, temp_l, test_size=0.50, random_state=42, stratify=temp_l
            )
        except ValueError:
            # Fallback if too few samples per class
            train_p, temp_p, train_l, temp_l = train_test_split(
                paths, labels, test_size=0.30, random_state=42
            )
            val_p, test_p, val_l, test_l = train_test_split(
                temp_p, temp_l, test_size=0.50, random_state=42
            )

        for p, l in zip(train_p, train_l):
            manifest_rows.append({"filepath": p, "modality": modality, "label": l, "is_valid": True, "split": "train"})
        for p, l in zip(val_p, val_l):
            manifest_rows.append({"filepath": p, "modality": modality, "label": l, "is_valid": True, "split": "validation"})
        for p, l in zip(test_p, test_l):
            manifest_rows.append({"filepath": p, "modality": modality, "label": l, "is_valid": True, "split": "test"})

    # Write manifest CSV
    manifest_csv_path = REPORTS_DIR / "dataset_manifest.csv"
    print(f"\nWriting dataset manifest to {manifest_csv_path}...")
    fieldnames = ["filepath", "modality", "label", "is_valid", "split"]
    with open(manifest_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    elapsed = time.time() - start_time
    print(f"Manifest creation complete in {elapsed:.1f}s.")
    print(f"Total indexed entries: {len(manifest_rows)}")

    # Print summary table
    df_manifest = pd.DataFrame(manifest_rows)
    print("\n================ DATASET MANIFEST SUMMARY ================")
    summary = df_manifest[df_manifest["is_valid"]].groupby(["modality", "split", "label"]).size().unstack(fill_value=0)
    print(summary)
    print("==========================================================")

if __name__ == "__main__":
    main()
