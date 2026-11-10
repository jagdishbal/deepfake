import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, precision_recall_curve, roc_auc_score, confusion_matrix

from preprocessing.clean_image import get_image_dataset

PROJECT_ROOT = Path(__file__).resolve().parent
VAL_DIR = PROJECT_ROOT / "dataset" / "image" / "validation"
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "saved_models" / "image" / "best_model_v2.h5"
CALIBRATION_FILE = PROJECT_ROOT / "models" / "calibration.json"

def calibrate_image_v2():
    print(f"Loading checkpoint with load_model: {CHECKPOINT_PATH}")
    model = tf.keras.models.load_model(str(CHECKPOINT_PATH), compile=False)

    print(f"Loading validation dataset from: {VAL_DIR}")
    val_ds = get_image_dataset(VAL_DIR, batch_size=32, shuffle=False)

    y_true = []
    y_probs = []

    for x, y in val_ds:
        preds = model.predict(x, verbose=0)
        y_true.extend(y.numpy().flatten())
        y_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    print(f"Validation samples: {len(y_true)} (Fake: {np.sum(y_true == 0)}, Real: {np.sum(y_true == 1)})")
    auc = roc_auc_score(y_true, y_probs)
    print(f"Validation ROC-AUC: {auc:.4f}")

    thresholds = np.arange(0.10, 0.90, 0.01)
    results = []

    best_thresh = 0.50
    best_macro_f1 = -1.0
    best_metrics = {}

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        acc = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec_real = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rec_real = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        prec_fake = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        rec_fake = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # Fake -> Real
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # Real -> Fake
        
        results.append({
            "threshold": float(t),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "fpr": float(fpr),
            "fnr": float(fnr),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        })

        # Objective: Highest Macro F1 with minimum distance to 0.50 for ties
        if macro_f1 > best_macro_f1 or (
            np.isclose(macro_f1, best_macro_f1, atol=1e-4) and abs(t - 0.50) < abs(best_thresh - 0.50)
        ):
            best_macro_f1 = macro_f1
            best_thresh = float(t)
            best_metrics = results[-1]

    print("\n" + "="*50)
    print("CALIBRATION RESULTS ON VALIDATION SPLIT")
    print("="*50)
    print(f"Optimal Threshold Selected: {best_thresh:.2f}")
    print(f"Validation Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Validation Macro F1: {best_metrics['macro_f1']:.4f}")
    print(f"False Positive Rate (Fake -> Real): {best_metrics['fpr']:.4f} ({best_metrics['fp']}/{best_metrics['fp']+best_metrics['tn']})")
    print(f"False Negative Rate (Real -> Fake): {best_metrics['fnr']:.4f} ({best_metrics['fn']}/{best_metrics['fn']+best_metrics['tp']})")
    print(f"Confusion Matrix [TN, FP / FN, TP]: [[{best_metrics['tn']}, {best_metrics['fp']}], [{best_metrics['fn']}, {best_metrics['tp']}]]")

    # Update calibration.json
    if CALIBRATION_FILE.exists():
        with open(CALIBRATION_FILE, "r") as f:
            calib_data = json.load(f)
    else:
        calib_data = {}

    calib_data["image"] = {
        "threshold": round(best_thresh, 2),
        "f1_score": float(best_macro_f1),
        "uncertainty_lower": round(max(0.0, best_thresh - 0.05), 2),
        "uncertainty_upper": round(min(1.0, best_thresh + 0.05), 2),
        "checkpoint": "models/saved_models/image/best_model_v2.h5"
    }

    with open(CALIBRATION_FILE, "w") as f:
        json.dump(calib_data, f, indent=4)

    print(f"\nUpdated {CALIBRATION_FILE} with image threshold {best_thresh:.2f}")
    return best_thresh, best_metrics

if __name__ == "__main__":
    calibrate_image_v2()
