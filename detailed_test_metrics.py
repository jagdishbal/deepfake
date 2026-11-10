import config
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from pathlib import Path
from preprocessing.clean_image import get_image_dataset

model = tf.keras.models.load_model("models/saved_models/image/best_model_v2.h5", compile=False)
threshold = 0.52

def evaluate_dataset(name, path):
    ds = get_image_dataset(path, batch_size=32, shuffle=False)
    y_true = []
    y_pred_probs = []
    
    for images, labels in ds:
        probs = model.predict(images, verbose=0).reshape(-1)
        y_true.extend(labels.numpy().reshape(-1))
        y_pred_probs.extend(probs)
        
    y_true = np.array(y_true, dtype=int)
    y_pred_probs = np.array(y_pred_probs, dtype=float)
    y_pred = (y_pred_probs >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1) # Precision for Real
    rec = recall_score(y_true, y_pred, pos_label=1)     # Recall for Real
    f1 = f1_score(y_true, y_pred, pos_label=1)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    auc = roc_auc_score(y_true, y_pred_probs)
    cm = confusion_matrix(y_true, y_pred) # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # Fake -> Real
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # Real -> Fake
    
    print(f"=== {name} ===")
    print(f"Path: {path}")
    print(f"Total Samples: {len(y_true)} (Fake/0={tn+fp}, Real/1={fn+tp})")
    print(f"Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precision (Real):{prec:.4f}")
    print(f"Recall (Real):   {rec:.4f}")
    print(f"F1-Score (Real): {f1:.4f}")
    print(f"Macro F1-Score:  {macro_f1:.4f}")
    print(f"ROC-AUC:         {auc:.4f}")
    print(f"Confusion Matrix: [[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]")
    print(f"FPR (Fake->Real):{fpr:.4f} ({fpr*100:.2f}%)")
    print(f"FNR (Real->Fake):{fnr:.4f} ({fnr*100:.2f}%)")
    print(f"Fake Detection Rate (Specificity): {tn/(tn+fp):.4f} ({tn/(tn+fp)*100:.2f}%)")
    print(f"Real Detection Rate (Sensitivity): {tp/(fn+tp):.4f} ({tp/(fn+tp)*100:.2f}%)\n")

evaluate_dataset("1. Untouched Independent Test Set", Path("dataset/image/test"))
evaluate_dataset("2. CIFAKE Independent Test Benchmark", Path("dataset/benchmark_cifake/test"))
