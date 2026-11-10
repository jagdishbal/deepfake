"""
Forensic analysis of the 13 misclassified videos (32.5% error rate on Fake videos).
Extracts detailed video metadata, frame visual properties, optical flow / motion,
and links back to FakeAVCeleb ground truth generator method.
"""

import csv
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
import predict

META_CSV = PROJECT_ROOT / "dataset" / "raw" / "audio" / "audio_dataset.zio" / "FakeAVCeleb_v1.2" / "FakeAVCeleb_v1.2" / "meta_data.csv"

def get_meta_map():
    meta_map = {}
    if META_CSV.exists():
        with open(META_CSV, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Key by filename
                fname = Path(row["path"]).name
                meta_map[fname] = row
    return meta_map

def inspect_video_properties(v_path: Path):
    cap = cv2.VideoCapture(str(v_path))
    if not cap.isOpened():
        return {"error": "cannot open"}
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Read first 10 frames and compute average brightness, contrast, and inter-frame motion
    frames = []
    for _ in range(min(15, total_frames)):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames.append(frame)
    cap.release()
    
    if not frames:
        return {"error": "no frames"}
        
    brightness = np.mean([np.mean(f) for f in frames])
    contrast = np.mean([np.std(f) for f in frames])
    
    # Motion between consecutive frames (mean absolute difference)
    motion = 0.0
    if len(frames) > 1:
        diffs = [np.mean(np.abs(frames[i].astype(float) - frames[i-1].astype(float))) for i in range(1, len(frames))]
        motion = np.mean(diffs)
        
    return {
        "fps": fps,
        "frames": total_frames,
        "duration": total_frames / max(1.0, fps),
        "resolution": f"{w}x{h}",
        "brightness": round(float(brightness), 2),
        "contrast": round(float(contrast), 2),
        "inter_frame_motion": round(float(motion), 2),
    }

def main():
    print("=" * 70)
    print("INVESTIGATION OF THE 13 MISCLASSIFIED VIDEO SAMPLES (32.5% FPR)")
    print("=" * 70)
    
    test_dir = PROJECT_ROOT / "dataset" / "video" / "test"
    predict._load_model.cache_clear()
    meta_map = get_meta_map()
    
    results = {"fake": [], "real": []}
    
    for cls in ["fake", "real"]:
        p = test_dir / cls
        for f in sorted(p.glob("*.mp4")):
            res = predict.predict_video(f, num_frames=10)
            p_real = res["probability_real"]
            pred = res["prediction"]
            is_correct = (pred.lower() == cls)
            entry = {
                "file": f.name,
                "true_cls": cls,
                "pred": pred,
                "prob_real": p_real,
                "prob_fake": res["probability_fake"],
                "is_inconclusive": res["is_inconclusive"],
                "confidence": res["confidence"],
                "is_correct": is_correct,
            }
            props = inspect_video_properties(f)
            entry.update(props)
            
            # Lookup metadata
            # File name pattern: {stem}_fake_{idx:03d}.mp4 -> stem could be 00004_id00495...
            clean_name = f.name.replace(f"_{cls}_", "_")
            # Search metadata
            match_row = None
            for meta_name, row in meta_map.items():
                if meta_name in f.name or f.stem.startswith(Path(meta_name).stem):
                    match_row = row
                    break
            entry["meta"] = match_row
            results[cls].append(entry)

    misclassified_fakes = [r for r in results["fake"] if not r["is_correct"]]
    correct_fakes = [r for r in results["fake"] if r["is_correct"]]
    
    print(f"\nTotal Fake test samples: {len(results['fake'])}")
    print(f"Correctly identified as Fake: {len(correct_fakes)} (67.5%)")
    print(f"Misclassified as Real (False Positives): {len(misclassified_fakes)} (32.5%)\n")
    
    print("-" * 70)
    print("DETAILS OF THE 13 MISCLASSIFIED FAKE VIDEOS:")
    print("-" * 70)
    for idx, r in enumerate(misclassified_fakes, 1):
        inconcl = "[INCONCLUSIVE]" if r["is_inconclusive"] else "[CONFIDENT WRONG]"
        meta_info = ""
        if r["meta"]:
            meta_info = f" | Type: {r['meta'].get('type')} | Method: {r['meta'].get('method')} | Race: {r['meta'].get('race')}"
        print(f"{idx:2d}. {r['file']}")
        print(f"    P(Real): {r['prob_real']*100:5.2f}% | Conf: {r['confidence']*100:5.2f}% | {inconcl}")
        print(f"    Resolution: {r.get('resolution')} | Motion: {r.get('inter_frame_motion')} | Brightness: {r.get('brightness')}{meta_info}")
        
    print("\n" + "-" * 70)
    print("COMPARATIVE CHARACTERISTIC ANALYSIS:")
    print("-" * 70)
    
    avg_motion_mis = np.mean([r["inter_frame_motion"] for r in misclassified_fakes])
    avg_motion_cor = np.mean([r["inter_frame_motion"] for r in correct_fakes])
    avg_bright_mis = np.mean([r["brightness"] for r in misclassified_fakes])
    avg_bright_cor = np.mean([r["brightness"] for r in correct_fakes])
    avg_contrast_mis = np.mean([r["contrast"] for r in misclassified_fakes])
    avg_contrast_cor = np.mean([r["contrast"] for r in correct_fakes])
    
    inconclusive_count = sum(1 for r in misclassified_fakes if r["is_inconclusive"])
    
    print(f"1. Inconclusive Boundary Rate:")
    print(f"   {inconclusive_count} out of {len(misclassified_fakes)} misclassified fakes were within the uncertainty boundary [0.22 - 0.38] and flagged INCONCLUSIVE.")
    print(f"2. Inter-Frame Motion:")
    print(f"   Misclassified Fakes avg motion: {avg_motion_mis:.2f}")
    print(f"   Correctly Detected Fakes avg motion: {avg_motion_cor:.2f}")
    print(f"3. Brightness / Illumination:")
    print(f"   Misclassified Fakes avg brightness: {avg_bright_mis:.2f}")
    print(f"   Correctly Detected Fakes avg brightness: {avg_bright_cor:.2f}")
    print(f"4. Contrast:")
    print(f"   Misclassified Fakes avg contrast: {avg_contrast_mis:.2f}")
    print(f"   Correctly Detected Fakes avg contrast: {avg_contrast_cor:.2f}")
    
    # Synthesis type breakdown
    methods = {}
    for r in misclassified_fakes:
        m = "Unknown"
        if "wavtolip" in r["file"].lower() or (r["meta"] and "wav2lip" in str(r["meta"]).lower()):
            m = "Wav2Lip (Lip-Sync only)"
        elif "faceswap" in r["file"].lower():
            m = "FaceSwap"
        elif "fsgan" in r["file"].lower():
            m = "FSGAN"
        methods[m] = methods.get(m, 0) + 1
        
    print(f"\n5. Synthesis Mechanism in Misclassified Fakes:")
    for m, c in methods.items():
        print(f"   {m}: {c} / {len(misclassified_fakes)} ({c/len(misclassified_fakes)*100:.1f}%)")

if __name__ == "__main__":
    main()
