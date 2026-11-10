"""
Train an Audio Deepfake Detection CNN on Mel-spectrograms.
Uses clean, non-contaminated real/fake speech samples from FakeAVCeleb with zero speaker leakage.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from preprocessing.clean_audio import extract_audio_features
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def load_audio_split(split_dir: Path):
    """Loads all audio files in a split directory and extracts mel-spectrogram features."""
    X = []
    y = []
    filepaths = []
    
    for label_idx, label_name in enumerate(["fake", "real"]):
        label_dir = split_dir / label_name
        files = sorted(list(label_dir.glob("*.wav")))
        for f in files:
            feat = extract_audio_features(str(f), sr=16000, duration=3.0, strict=False)
            X.append(feat)
            y.append(label_idx)
            filepaths.append(str(f))
            
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), filepaths


def build_enhanced_audio_model(input_shape=(128, None, 1), learning_rate=3e-4):
    """
    Constructs a 4-stage convolutional neural network with batch normalization
    and global average pooling for audio deepfake classification.
    0 = Fake, 1 = Real.
    """
    inputs = layers.Input(shape=input_shape)

    # Conv Block 1
    x = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Conv Block 4
    x = layers.Conv2D(256, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.GlobalAveragePooling2D()(x)

    # Classifier Head
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="AudioDeepfakeDetector_V2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    return model


def main():
    print("=" * 60)
    print("STARTING AUDIO MODEL TRAINING (V2)")
    print("=" * 60)

    dataset_dir = PROJECT_ROOT / "dataset" / "audio"
    models_dir = PROJECT_ROOT / "models" / "saved_models" / "audio"
    logs_dir = PROJECT_ROOT / "logs" / "audio"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data into memory (700 files * 128*94*1*4 bytes ~= 33 MB RAM)
    print("Loading train split...")
    X_train, y_train, _ = load_audio_split(dataset_dir / "train")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}, real={int(sum(y_train))}, fake={int(len(y_train)-sum(y_train))}")

    print("Loading validation split...")
    X_val, y_val, _ = load_audio_split(dataset_dir / "validation")
    print(f"  Validation: X={X_val.shape}, y={y_val.shape}, real={int(sum(y_val))}, fake={int(len(y_val)-sum(y_val))}")

    print("Loading test split...")
    X_test, y_test, test_paths = load_audio_split(dataset_dir / "test")
    print(f"  Test: X={X_test.shape}, y={y_test.shape}, real={int(sum(y_test))}, fake={int(len(y_test)-sum(y_test))}")

    # 2. Build model
    model = build_enhanced_audio_model(input_shape=(128, X_train.shape[2], 1), learning_rate=3e-4)
    model.summary()

    # 3. Callbacks
    checkpoint_v2 = models_dir / "best_model_v2.h5"
    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_v2),
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        callbacks.CSVLogger(str(logs_dir / "training_log_v2.csv")),
    ]

    # 4. Train
    print("\nTraining Audio Deepfake Detector...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=20,
        batch_size=32,
        shuffle=True,
        callbacks=cb_list,
    )

    # 5. Evaluate on Validation Set for Threshold Calibration
    print("\n" + "=" * 60)
    print("CALIBRATING DECISION THRESHOLD ON VALIDATION SET")
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

    # 6. Final Evaluation on Independent Test Set
    print("\n" + "=" * 60)
    print("EVALUATING MODEL ON UNSEEN TEST SET")
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

    # If accuracy and AUC exceed 80%, update production best_model.h5
    if acc > 0.80 and auc > 0.85:
        import shutil
        prod_model = models_dir / "best_model.h5"
        shutil.copy2(checkpoint_v2, prod_model)
        print(f"\nModel promoted to active production checkpoint: {prod_model}")

        # Update calibration.json
        import json
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

    print("\nAudio Model Training & Validation Complete!")


if __name__ == "__main__":
    main()
