import argparse
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, log_loss, roc_auc_score

from preprocessing.clean_image import get_image_dataset
from preprocessing.clean_video import get_video_dataset
from preprocessing.clean_audio import get_audio_dataset
from predict import _load_model, get_threshold

def evaluate_modality(modality, batch_size=32):
    project_root = Path(__file__).parent.resolve()
    dataset_dir = project_root / "dataset" / modality
    models_dir = project_root / "models" / "saved_models" / modality
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = models_dir / "best_model.h5"
    if not checkpoint_path.exists():
        print(f"Error: No trained model found at {checkpoint_path}")
        return

    test_dir = dataset_dir / "test"
    if not test_dir.exists():
        print(f"Error: Test dataset not found at {test_dir}")
        return

    print(f"=== Evaluating {modality.upper()} ===")

    if modality == 'image':
        test_ds = get_image_dataset(test_dir, batch_size=batch_size, subset="validation", shuffle=False)
    elif modality == 'video':
        video_batch = min(batch_size, 4)
        test_ds = get_video_dataset(test_dir, batch_size=video_batch, shuffle=False)
    elif modality == 'audio':
        test_ds = get_audio_dataset(test_dir, batch_size=batch_size, shuffle=False)
    else:
        raise ValueError(f"Invalid modality: {modality}")

    model = _load_model(modality)
    threshold = get_threshold(modality)

    y_true = []
    y_pred_probs = []

    print("Running inference on test set...")
    for x, y in test_ds:
        preds = model.predict(x, verbose=0)
        y_true.extend(y.numpy().flatten())
        y_pred_probs.extend(preds.flatten())

    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    y_pred_classes = (y_pred_probs >= threshold).astype(int)

    if len(y_true) == 0:
        print("No test data found.")
        return

    # Calculate metrics
    report = classification_report(y_true, y_pred_classes, target_names=['fake', 'real'], output_dict=True, zero_division=0)
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        roc_auc = 0.0 # Only one class in true labels

    loss = log_loss(y_true, y_pred_probs, labels=[0, 1])

    # Print and save report
    report_text = f"EVALUATION REPORT: {modality.upper()}\n"
    report_text += "="*40 + "\n"
    report_text += f"Loss: {loss:.4f}\n"
    report_text += f"Accuracy at threshold {threshold:.2f}: {accuracy_score(y_true, y_pred_classes):.4f}\n"
    report_text += f"ROC-AUC Score: {roc_auc:.4f}\n\n"
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_classes, labels=[0, 1]).ravel()
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0
    false_negative_rate = fn / (fn + tp) if fn + tp else 0.0
    report_text += f"False Positive Rate (Fake -> Real): {false_positive_rate:.4f}\n"
    report_text += f"False Negative Rate (Real -> Fake): {false_negative_rate:.4f}\n\n"
    report_text += "Classification Report:\n"
    report_text += classification_report(y_true, y_pred_classes, target_names=['fake', 'real'], zero_division=0) + "\n"
    report_text += "Confusion Matrix:\n"
    report_text += str(conf_matrix) + "\n"

    print(report_text)
    
    report_file = reports_dir / f"{modality}_evaluation_report.txt"
    with open(report_file, "w") as f:
        f.write(report_text)
    
    print(f"Report saved to {report_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Models")
    parser.add_argument("--modality", choices=["image", "video", "audio"], required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    evaluate_modality(args.modality, args.batch_size)
