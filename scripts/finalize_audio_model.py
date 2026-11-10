"""
Train, Calibrate, and Finalize the Production Audio Deepfake Detection Model.
Saves model with baked-in Normalization layer into models/saved_models/audio/best_model_v2.h5
and promotes to best_model.h5.
Updates models/calibration.json.
"""

import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
import librosa
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def extract_acoustic_features(audio_path: str | Path, sr: int = 16000, duration: float = 3.0) -> np.ndarray:
    """Extracts 140 comprehensive acoustic features."""
    y, _ = librosa.load(str(audio_path), sr=sr, duration=duration)
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    feats = np.hstack([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1),
        np.mean(delta2, axis=1), np.std(delta2, axis=1),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        np.mean(rolloff), np.std(rolloff),
        np.mean(flatness), np.std(flatness),
        np.mean(zcr), np.std(zcr),
    ])
    return feats.astype(np.float32)


def load_split(split_dir: Path):
    X, y = [], []
    for label_idx, label_name in enumerate(["fake", "real"]):
        p = split_dir / label_name
        files = sorted(list(p.glob("*.wav")))
        for f in files:
            X.append(extract_acoustic_features(f))
            y.append(label_idx)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def main():
    print("=" * 60)
    print("FINALIZING PRODUCTION AUDIO DEEPFAKE MODEL")
    print("=" * 60)

    dataset_dir = PROJECT_ROOT / "dataset" / "audio"
    models_dir = PROJECT_ROOT / "models" / "saved_models" / "audio"
    logs_dir = PROJECT_ROOT / "logs" / "audio"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Train split (700 clips)...")
    X_train, y_train = load_split(dataset_dir / "train")
    print(f"  Train: shape={X_train.shape}, real={int(sum(y_train))}, fake={int(len(y_train)-sum(y_train))}")

    print("Loading Validation split (150 clips)...")
    X_val, y_val = load_split(dataset_dir / "validation")
    print(f"  Val: shape={X_val.shape}, real={int(sum(y_val))}, fake={int(len(y_val)-sum(y_val))}")

    print("Loading Test split (150 clips)...")
    X_test, y_test = load_split(dataset_dir / "test")
    print(f"  Test: shape={X_test.shape}, real={int(sum(y_test))}, fake={int(len(y_test)-sum(y_test))}")

    # Build Keras model with internal Normalization layer
    norm_layer = tf.keras.layers.Normalization(axis=-1, name="acoustic_norm")
    norm_layer.adapt(X_train)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(140,), name="acoustic_features_input"),
        norm_layer,
        tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid", name="probability_real_output"),
    ], name="AudioDeepfakeDetector_V2")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    model.summary()

    cb_list = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=12, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.CSVLogger(str(logs_dir / "training_log_final.csv")),
    ]

    print("\nTraining Model...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=60,
        batch_size=32,
        shuffle=True,
        callbacks=cb_list,
    )

    # Threshold Calibration on Validation Set
    print("\n" + "=" * 60)
    print("CALIBRATING AUDIO DECISION THRESHOLD ON VALIDATION SET")
    print("=" * 60)
    val_probs = model.predict(X_val, verbose=0).flatten()

    best_thresh = 0.50
    best_f1 = 0.0
    for t in np.arange(0.30, 0.70, 0.01):
        preds = (val_probs >= t).astype(int)
        f1 = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = round(float(t), 2)

    val_preds = (val_probs >= best_thresh).astype(int)
    val_acc = accuracy_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Optimal Threshold: {best_thresh:.2f}")
    print(f"Validation Accuracy: {val_acc:.4f}, F1: {best_f1:.4f}, ROC-AUC: {val_auc:.4f}")

    # Final Evaluation on Independent Test Set
    print("\n" + "=" * 60)
    print("EVALUATING AUDIO MODEL ON UNSEEN TEST SET")
    print("=" * 60)
    test_probs = model.predict(X_test, verbose=0).flatten()
    test_preds = (test_probs >= best_thresh).astype(int)

    cm = confusion_matrix(y_test, test_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)
    auc = roc_auc_score(y_test, test_probs)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    print(f"Test Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Test Precision: {prec:.4f}")
    print(f"Test Recall:    {rec:.4f}")
    print(f"Test F1-score:  {f1:.4f}")
    print(f"Test ROC-AUC:   {auc:.4f}")
    print(f"Test FPR:       {fpr:.4f}")
    print(f"Test FNR:       {fnr:.4f}")
    print(f"Confusion Matrix (TN, FP / FN, TP):\n{cm}")
    print(f"Fake Prob Dist: mean={test_probs[y_test==0].mean():.4f}, std={test_probs[y_test==0].std():.4f}, range=[{test_probs[y_test==0].min():.4f}, {test_probs[y_test==0].max():.4f}]")
    print(f"Real Prob Dist: mean={test_probs[y_test==1].mean():.4f}, std={test_probs[y_test==1].std():.4f}, range=[{test_probs[y_test==1].min():.4f}, {test_probs[y_test==1].max():.4f}]")

    # Save models
    v2_checkpoint = models_dir / "best_model_v2.h5"
    model.save(str(v2_checkpoint))
    print(f"\nSaved best_model_v2.h5 to {v2_checkpoint}")

    prod_model = models_dir / "best_model.h5"
    shutil.copy2(v2_checkpoint, prod_model)
    print(f"Promoted to active production model: {prod_model}")

    # Update calibration.json
    calib_file = PROJECT_ROOT / "models" / "calibration.json"
    with open(calib_file, "r", encoding="utf-8") as f:
        calib_data = json.load(f)

    calib_data["audio"] = {
        "threshold": best_thresh,
        "f1_score": float(f1),
        "accuracy": float(acc),
        "roc_auc": float(auc),
        "uncertainty_lower": round(best_thresh - 0.08, 2),
        "uncertainty_upper": round(best_thresh + 0.08, 2),
        "checkpoint": "models/saved_models/audio/best_model_v2.h5",
    }
    with open(calib_file, "w", encoding="utf-8") as f:
        json.dump(calib_data, f, indent=4)
    print(f"Calibration updated in {calib_file}")

    print("\nAudio Model Finalization Complete!")


if __name__ == "__main__":
    main()
