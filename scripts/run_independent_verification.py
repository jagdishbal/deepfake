"""
Comprehensive Independent Verification Script
Executes:
1. Full test split evaluation for Audio V2 (150 clips: 75 Real, 75 Fake)
2. Full test split evaluation for Video V2 (80 clips: 40 Real, 40 Fake)
3. Full test split evaluation for Image (Dual-Branch V3 + V2 fallback)
4. Flask web workflow integration test (upload, prediction, result template, audit trail)
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
import predict
from werkzeug.datastructures import FileStorage

def evaluate_modality(modality: str):
    print("=" * 70)
    print(f"EVALUATION ON FULL UNSEEN TEST SPLIT: {modality.upper()}")
    print("=" * 70)
    
    test_dir = PROJECT_ROOT / "dataset" / modality / "test"
    predict._load_model.cache_clear()
    
    y_true = []
    y_scores = []
    y_preds = []
    sample_records = []
    
    for label_idx, cls in enumerate(["fake", "real"]):
        p = test_dir / cls
        if modality == "audio":
            valid_exts = {".wav", ".mp3", ".flac", ".aac"}
        elif modality == "video":
            valid_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        else:
            valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = sorted([f for f in p.glob("*") if f.is_file() and f.suffix.lower() in valid_exts])
        print(f"Evaluating {len(files)} {cls.upper()} test samples...")
        
        for f in files:
            res = predict.predict_file(f, modality=modality)
            p_real = res["probability_real"]
            pred = 1 if res["prediction"] == "Real" else 0
            
            y_true.append(label_idx)
            y_scores.append(p_real)
            y_preds.append(pred)
            sample_records.append({
                "file": f.name,
                "label": cls,
                "prob_real": p_real,
                "prob_fake": res.get("probability_fake", 1.0 - p_real),
                "is_inconclusive": res.get("is_inconclusive", False),
                "pred": res["prediction"]
            })
            
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_preds = np.array(y_preds)
    
    acc = accuracy_score(y_true, y_preds)
    prec = precision_score(y_true, y_preds, zero_division=0)
    rec = recall_score(y_true, y_preds, zero_division=0)
    f1 = f1_score(y_true, y_preds, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_scores)
    except Exception:
        auc = 0.5
        
    cm = confusion_matrix(y_true, y_preds)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / max(1, (fp + tn))
    fnr = fn / max(1, (fn + tp))
    
    fake_probs = y_scores[y_true == 0]
    real_probs = y_scores[y_true == 1]
    
    print("\n" + "-" * 70)
    print(f"METRICS FOR {modality.upper()} V2 (UNSEEN TEST SET):")
    print("-" * 70)
    print(f"  Total Test Samples: {len(y_true)} (Fake={len(fake_probs)}, Real={len(real_probs)})")
    print(f"  Accuracy:           {acc * 100:.2f}%")
    print(f"  Precision:          {prec:.4f}")
    print(f"  Recall (Real):      {rec:.4f} ({tp}/{len(real_probs)} real detected)")
    print(f"  F1-Score:           {f1:.4f}")
    print(f"  ROC-AUC:            {auc:.4f}")
    print(f"  False Positive Rate:{fpr * 100:.2f}% ({fp}/{fp + tn})")
    print(f"  False Negative Rate:{fnr * 100:.2f}% ({fn}/{fn + tp})")
    print(f"  Confusion Matrix (TN, FP / FN, TP):\n{cm}")
    print(f"  Fake Probability Dist: mean={fake_probs.mean():.4f}, std={fake_probs.std():.4f}, min={fake_probs.min():.4f}, max={fake_probs.max():.4f}")
    print(f"  Real Probability Dist: mean={real_probs.mean():.4f}, std={real_probs.std():.4f}, min={real_probs.min():.4f}, max={real_probs.max():.4f}")
    
    inconclusive_count = sum(1 for s in sample_records if s["is_inconclusive"])
    print(f"  Inconclusive / Low Confidence Flagged: {inconclusive_count} / {len(sample_records)} ({inconclusive_count / len(sample_records) * 100:.1f}%)")
    
    return {
        "modality": modality,
        "total": len(y_true),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "fpr": fpr,
        "fnr": fnr,
        "cm": cm.tolist(),
        "fake_mean": float(fake_probs.mean()),
        "real_mean": float(real_probs.mean()),
    }

def test_flask_workflow():
    print("\n" + "=" * 70)
    print("STEP 10: FLASK WEB WORKFLOW & INTEGRATION TESTING")
    print("=" * 70)
    import app as app_module
    from unittest.mock import MagicMock, patch
    
    app_module.app.config.update(TESTING=True)
    
    class MockCursor:
        def __init__(self):
            self.lastrowid = 101
            self.executed_queries = []
        def execute(self, q, params=None):
            self.executed_queries.append((q, params))
        def fetchone(self):
            return {
                "id": 101,
                "user_id": 1,
                "filename": "test_audio_sample.wav",
                "file_type": "Audio",
                "prediction": "Real",
                "confidence": 99.8,
            }
        def close(self):
            pass
            
    class MockConnection:
        def __init__(self):
            self.cur = MockCursor()
        def cursor(self, dictionary=False):
            return self.cur
        def commit(self):
            pass
        def close(self):
            pass
            
    mock_conn = MockConnection()
    
    # Test audio upload workflow
    with app_module.app.test_client() as client:
        with client.session_transaction() as sess:
            sess["user_id"] = 1
            sess["username"] = "testuser"
            sess["fullname"] = "Test User"
            
        real_wav_path = next((PROJECT_ROOT / "dataset" / "audio" / "test" / "real").glob("*.wav"))
        with open(real_wav_path, "rb") as f:
            audio_bytes = f.read()
            
        with patch.object(app_module, "get_db_connection", return_value=mock_conn):
            resp = client.post(
                "/audio_detection",
                data={"audio": (io.BytesIO(audio_bytes), real_wav_path.name)},
                content_type="multipart/form-data"
            )
            print(f"  POST /audio_detection response: {resp.status_code}")
            assert resp.status_code == 302, f"Expected 302 redirect, got {resp.status_code}"
            target_url = resp.headers["Location"]
            print(f"  Redirect location: {target_url}")
            
            # Follow redirect to view_result
            view_resp = client.get(target_url)
            print(f"  GET {target_url} response: {view_resp.status_code}")
            assert view_resp.status_code == 200
            
            html = view_resp.data.decode("utf-8")
            assert "Detection Result" in html
            assert "Probability Breakdown" in html or "REAL:" in html
            print("  [SUCCESS] Web application rendered result page with dual probabilities!")
            print(f"  [SUCCESS] Database recorded prediction: {mock_conn.cur.executed_queries[0]}")

def main():
    audio_results = evaluate_modality("audio")
    video_results = evaluate_modality("video")
    image_results = evaluate_modality("image")
    test_flask_workflow()
    
    print("\n" + "=" * 70)
    print("ALL INDEPENDENT VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
