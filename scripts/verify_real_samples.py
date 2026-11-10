"""
Verify real audio, video, and image samples using active production models.
Produces comprehensive metrics, confusion matrices, and individual sample traces.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
import numpy as np
import predict

def test_modality(modality: str, num_samples_per_class: int = 10):
    print("=" * 60)
    print(f"TESTING REAL SAMPLES FOR MODALITY: {modality.upper()}")
    print("=" * 60)
    predict._load_model.cache_clear()
    
    test_dir = PROJECT_ROOT / "dataset" / modality / "test"
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return
    
    results = {"real": [], "fake": []}
    
    for cls in ["fake", "real"]:
        cls_dir = test_dir / cls
        files = sorted(list(cls_dir.glob("*.wav" if modality == "audio" else "*.mp4" if modality == "video" else "*.jpg")))
        samples = files[:num_samples_per_class]
        print(f"\nProcessing {len(samples)} {cls.upper()} samples:")
        
        for f in samples:
            try:
                res = predict.predict_file(f, modality=modality)
                pred = res["prediction"]
                conf = res["confidence"]
                p_real = res["probability_real"]
                p_fake = res.get("probability_fake", 1.0 - p_real)
                inconclusive = res.get("is_inconclusive", False)
                status = res.get("confidence_status", "CONFIDENT")
                
                is_correct = (pred.lower() == cls)
                mark = "[CORRECT]" if is_correct else "[WRONG]"
                
                print(f"  {mark} {f.name[:35]:35s} -> Pred: {pred:4s} | Conf: {conf*100:5.1f}% | P(Real): {p_real*100:5.1f}% | P(Fake): {p_fake*100:5.1f}% | Inconclusive: {inconclusive}")
                results[cls].append({
                    "file": f.name,
                    "true_label": cls,
                    "pred_label": pred.lower(),
                    "confidence": conf,
                    "probability_real": p_real,
                    "probability_fake": p_fake,
                    "is_inconclusive": inconclusive,
                    "is_correct": is_correct,
                })
            except Exception as exc:
                print(f"  [ERROR] {f.name}: {exc}")
                
    total_samples = len(results["fake"]) + len(results["real"])
    correct_samples = sum(r["is_correct"] for r in results["fake"] + results["real"])
    acc = correct_samples / total_samples if total_samples > 0 else 0.0
    
    tp = sum(1 for r in results["real"] if r["is_correct"])
    fn = sum(1 for r in results["real"] if not r["is_correct"])
    tn = sum(1 for r in results["fake"] if r["is_correct"])
    fp = sum(1 for r in results["fake"] if not r["is_correct"])
    
    print(f"\nSummary for {modality.upper()}:")
    print(f"  Total Evaluated: {total_samples}")
    print(f"  Correct:         {correct_samples} ({acc*100:.2f}%)")
    print(f"  Confusion Matrix:")
    print(f"    TN (Fake correct): {tn} | FP (Fake as Real): {fp}")
    print(f"    FN (Real as Fake): {fn} | TP (Real correct): {tp}")
    if results["real"]:
        print(f"  Real Probability Range: [{min(r['probability_real'] for r in results['real']):.4f}, {max(r['probability_real'] for r in results['real']):.4f}]")
    if results["fake"]:
        print(f"  Fake Probability Range: [{min(r['probability_real'] for r in results['fake']):.4f}, {max(r['probability_real'] for r in results['fake']):.4f}]")
    return results

def main():
    test_modality("audio", num_samples_per_class=10)
    test_modality("video", num_samples_per_class=10)
    test_modality("image", num_samples_per_class=10)

if __name__ == "__main__":
    main()
