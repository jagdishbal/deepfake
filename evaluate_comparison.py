import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss

from preprocessing.clean_image import get_image_dataset
from models.image_model import build_image_model

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "dataset" / "image" / "test"
CIFAKE_TEST_DIR = PROJECT_ROOT / "dataset" / "benchmark_cifake" / "test"
REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

ORIGINAL_CHECKPOINT = PROJECT_ROOT / "models" / "saved_models" / "image" / "best_model.h5"
NEW_CHECKPOINT = PROJECT_ROOT / "models" / "saved_models" / "image" / "best_model_v2.h5"

def run_evaluation_on_dataset(model, dataset_dir, threshold, model_name="Model", dataset_name="Test Set"):
    test_ds = get_image_dataset(dataset_dir, batch_size=32, shuffle=False)
    
    y_true = []
    y_pred_probs = []

    for x, y in test_ds:
        preds = model.predict(x, verbose=0)
        y_true.extend(y.numpy().flatten())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_classes = (y_pred_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred_classes)
    auc = roc_auc_score(y_true, y_pred_probs)
    loss = log_loss(y_true, y_pred_probs, labels=[0, 1])

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_classes, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # Fake -> Real
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # Real -> Fake

    clf_report = classification_report(y_true, y_pred_classes, target_names=['Fake (0)', 'Real (1)'], output_dict=True, zero_division=0)
    conf_matrix = [[int(tn), int(fp)], [int(fn), int(tp)]]

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "threshold": float(threshold),
        "total": len(y_true),
        "fake_total": int(np.sum(y_true == 0)),
        "real_total": int(np.sum(y_true == 1)),
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "loss": float(loss),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "report_dict": clf_report,
        "confusion_matrix": conf_matrix
    }

def main():
    print("="*60)
    print("COMPARATIVE EVALUATION: ORIGINAL MODEL vs NEW MODEL V2")
    print("="*60)

    # 1. Load Original Model
    orig_model = build_image_model(weights=None)
    orig_model.load_weights(str(ORIGINAL_CHECKPOINT))
    orig_thresh = 0.59

    # 2. Load New Model V2
    new_model = tf.keras.models.load_model(str(NEW_CHECKPOINT), compile=False)
    new_thresh = 0.52

    # A. Untouched original test dataset (231 images)
    res_orig_test = run_evaluation_on_dataset(orig_model, TEST_DIR, orig_thresh, "Original Model", "Untouched Test Set")
    res_new_test = run_evaluation_on_dataset(new_model, TEST_DIR, new_thresh, "New Model V2", "Untouched Test Set")

    # B. AI-Generated Benchmark test dataset (180 images)
    res_orig_cifake = run_evaluation_on_dataset(orig_model, CIFAKE_TEST_DIR, orig_thresh, "Original Model", "CIFAKE Synthetic Benchmark")
    res_new_cifake = run_evaluation_on_dataset(new_model, CIFAKE_TEST_DIR, new_thresh, "New Model V2", "CIFAKE Synthetic Benchmark")

    # Print Report
    report_text = "# Image Model Comprehensive Evaluation & Comparison Report\n\n"
    report_text += "## 1. Primary Benchmark: Untouched Independent Test Set (N=231)\n"
    report_text += f"- Ground Truth: {res_orig_test['fake_total']} Fake (Video Manipulation) / {res_orig_test['real_total']} Real\n\n"
    report_text += "| Metric | Original Model | New Model V2 | Change |\n"
    report_text += "| :--- | :---: | :---: | :---: |\n"
    report_text += f"| Decision Threshold | {res_orig_test['threshold']:.2f} | {res_new_test['threshold']:.2f} | -0.07 |\n"
    report_text += f"| Accuracy | {res_orig_test['accuracy']*100:.2f}% | {res_new_test['accuracy']*100:.2f}% | {(res_new_test['accuracy']-res_orig_test['accuracy'])*100:+.2f}% |\n"
    report_text += f"| ROC-AUC Score | {res_orig_test['roc_auc']:.4f} | {res_new_test['roc_auc']:.4f} | {res_new_test['roc_auc']-res_orig_test['roc_auc']:+.4f} |\n"
    report_text += f"| Loss (Binary Crossentropy) | {res_orig_test['loss']:.4f} | {res_new_test['loss']:.4f} | {res_new_test['loss']-res_orig_test['loss']:+.4f} |\n"
    report_text += f"| Real Images Correct (TP) | {res_orig_test['tp']} / {res_orig_test['real_total']} ({res_orig_test['tp']/res_orig_test['real_total']*100:.1f}%) | {res_new_test['tp']} / {res_new_test['real_total']} ({res_new_test['tp']/res_new_test['real_total']*100:.1f}%) | {res_new_test['tp']-res_orig_test['tp']:+d} |\n"
    report_text += f"| Real Misclassified as Fake (FN) | {res_orig_test['fn']} / {res_orig_test['real_total']} ({res_orig_test['fnr']*100:.1f}%) | {res_new_test['fn']} / {res_new_test['real_total']} ({res_new_test['fnr']*100:.1f}%) | {res_new_test['fn']-res_orig_test['fn']:+d} (Reduced!) |\n"
    report_text += f"| Fake Images Correct (TN) | {res_orig_test['tn']} / {res_orig_test['fake_total']} ({res_orig_test['tn']/res_orig_test['fake_total']*100:.1f}%) | {res_new_test['tn']} / {res_new_test['fake_total']} ({res_new_test['tn']/res_new_test['fake_total']*100:.1f}%) | {res_new_test['tn']-res_orig_test['tn']:+d} |\n"
    report_text += f"| Fake Misclassified as Real (FP) | {res_orig_test['fp']} / {res_orig_test['fake_total']} ({res_orig_test['fpr']*100:.1f}%) | {res_new_test['fp']} / {res_new_test['fake_total']} ({res_new_test['fpr']*100:.1f}%) | {res_new_test['fp']-res_orig_test['fp']:+d} |\n"
    report_text += f"| False Positive Rate (Fake -> Real) | {res_orig_test['fpr']*100:.2f}% | {res_new_test['fpr']*100:.2f}% | {(res_new_test['fpr']-res_orig_test['fpr'])*100:+.2f}% |\n"
    report_text += f"| False Negative Rate (Real -> Fake) | {res_orig_test['fnr']*100:.2f}% | {res_new_test['fnr']*100:.2f}% | {(res_new_test['fnr']-res_orig_test['fnr'])*100:+.2f}% |\n\n"

    report_text += "### Confusion Matrices:\n"
    report_text += f"- **Original Model Confusion Matrix:** `[[{res_orig_test['tn']}, {res_orig_test['fp']}], [{res_orig_test['fn']}, {res_orig_test['tp']}]]`\n"
    report_text += f"- **New Model V2 Confusion Matrix:** `[[{res_new_test['tn']}, {res_new_test['fp']}], [{res_new_test['fn']}, {res_new_test['tp']}]]`\n\n"

    report_text += "## 2. AI-Generated Synthetic Media Benchmark: CIFAKE Test Split (N=180)\n"
    report_text += f"- Ground Truth: {res_orig_cifake['fake_total']} AI-Generated (Stable Diffusion) / {res_orig_cifake['real_total']} Authentic Photos\n\n"
    report_text += "| Metric | Original Model | New Model V2 | Improvement |\n"
    report_text += "| :--- | :---: | :---: | :---: |\n"
    report_text += f"| Accuracy | {res_orig_cifake['accuracy']*100:.2f}% | {res_new_cifake['accuracy']*100:.2f}% | {(res_new_cifake['accuracy']-res_orig_cifake['accuracy'])*100:+.2f}% |\n"
    report_text += f"| ROC-AUC Score | {res_orig_cifake['roc_auc']:.4f} | {res_new_cifake['roc_auc']:.4f} | {res_new_cifake['roc_auc']-res_orig_cifake['roc_auc']:+.4f} |\n"
    report_text += f"| AI-Generated Correctly Detected | {res_orig_cifake['tn']} / {res_orig_cifake['fake_total']} ({res_orig_cifake['tn']/res_orig_cifake['fake_total']*100:.1f}%) | {res_new_cifake['tn']} / {res_new_cifake['fake_total']} ({res_new_cifake['tn']/res_new_cifake['fake_total']*100:.1f}%) | {res_new_cifake['tn']-res_orig_cifake['tn']:+d} |\n"
    report_text += f"| AI-Generated Misclassified as Real | {res_orig_cifake['fp']} / {res_orig_cifake['fake_total']} ({res_orig_cifake['fpr']*100:.1f}%) | {res_new_cifake['fp']} / {res_new_cifake['fake_total']} ({res_new_cifake['fpr']*100:.1f}%) | {res_new_cifake['fp']-res_orig_cifake['fp']:+d} |\n\n"

    print(report_text)

    out_file = REPORTS_DIR / "image_v2_evaluation_report.txt"
    with open(out_file, "w") as f:
        f.write(report_text)
    print(f"Report saved to: {out_file}")

if __name__ == "__main__":
    main()
