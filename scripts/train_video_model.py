"""
Train a Video Deepfake Detection Model using a fine-tuned MobileNetV2 spatial backbone
and a Bidirectional GRU temporal aggregation head.
Uses clean, non-contaminated real and fake video clips from FakeAVCeleb with zero speaker leakage.
Produces an end-to-end unified Keras model matching the exact (None, num_frames, 224, 224, 3) signature.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


NUM_FRAMES = 5
IMAGE_SIZE = (224, 224)


def extract_frames_for_video(video_path: Path, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """Extracts uniform frames from video and applies MobileNetV2 normalization."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = num_frames
    indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            frames.append(np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, IMAGE_SIZE)
        frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame.astype(np.float32))
        frames.append(frame)
    cap.release()
    return np.array(frames, dtype=np.float32)


def main():
    print("=" * 60)
    print("STARTING VIDEO MODEL TRAINING (V2)")
    print("=" * 60)

    dataset_dir = PROJECT_ROOT / "dataset" / "video"
    models_dir = PROJECT_ROOT / "models" / "saved_models" / "video"
    logs_dir = PROJECT_ROOT / "logs" / "video"
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load fine-tuned MobileNetV2 spatial backbone
    v2_img_path = PROJECT_ROOT / "models" / "saved_models" / "image" / "best_model_v2.h5"
    if v2_img_path.exists():
        print(f"Loading deepfake-adapted spatial backbone from {v2_img_path}...")
        v2_img_model = tf.keras.models.load_model(str(v2_img_path), compile=False)
        base_mobilenet = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights=None
        )
        base_mobilenet.set_weights(v2_img_model.layers[0].get_weights())
    else:
        print("Using ImageNet MobileNetV2 backbone...")
        base_mobilenet = tf.keras.applications.MobileNetV2(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet"
        )

    spatial_extractor = tf.keras.Sequential([
        base_mobilenet,
        layers.GlobalAveragePooling2D(),
    ])

    # 2. Extract frame embeddings for all splits
    def load_split_embeddings(split_name: str):
        print(f"Extracting embeddings for {split_name} split...")
        X_embs = []
        y_labels = []
        for label_idx, label_name in enumerate(["fake", "real"]):
            p = dataset_dir / split_name / label_name
            files = sorted(list(p.glob("*.mp4")))
            for f in files:
                frames = extract_frames_for_video(f, num_frames=NUM_FRAMES)
                # Compute (num_frames, 1280) embedding
                emb = spatial_extractor.predict(frames, verbose=0)
                X_embs.append(emb)
                y_labels.append(label_idx)
        return np.array(X_embs, dtype=np.float32), np.array(y_labels, dtype=np.float32)

    X_train_emb, y_train = load_split_embeddings("train")
    X_val_emb, y_val = load_split_embeddings("validation")
    X_test_emb, y_test = load_split_embeddings("test")

    print(f"Train embeddings: {X_train_emb.shape}, labels: {y_train.shape}")
    print(f"Val embeddings:   {X_val_emb.shape}, labels: {y_val.shape}")
    print(f"Test embeddings:  {X_test_emb.shape}, labels: {y_test.shape}")

    # 3. Train Temporal Sequence Model (Bidirectional GRU)
    print("\nBuilding Temporal Classifier...")
    temporal_input = layers.Input(shape=(NUM_FRAMES, 1280))
    x = layers.Bidirectional(layers.GRU(32, dropout=0.3, return_sequences=False))(temporal_input)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    temporal_output = layers.Dense(1, activation="sigmoid")(x)

    temporal_model = models.Model(inputs=temporal_input, outputs=temporal_output, name="TemporalHead")
    temporal_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
    )
    temporal_model.summary()

    cb_list = [
        callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1),
        callbacks.CSVLogger(str(logs_dir / "training_log_v2.csv")),
    ]

    print("\nTraining Temporal Model on embeddings...")
    temporal_model.fit(
        X_train_emb,
        y_train,
        validation_data=(X_val_emb, y_val),
        epochs=30,
        batch_size=16,
        shuffle=True,
        callbacks=cb_list,
    )

    # 4. Calibrate Threshold on Validation Set
    print("\n" + "=" * 60)
    print("CALIBRATING VIDEO DECISION THRESHOLD ON VALIDATION SET")
    print("=" * 60)
    val_probs = temporal_model.predict(X_val_emb, verbose=0).flatten()

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

    # 5. Evaluate on Independent Test Set
    print("\n" + "=" * 60)
    print("EVALUATING MODEL ON UNSEEN TEST SET")
    print("=" * 60)
    test_probs = temporal_model.predict(X_test_emb, verbose=0).flatten()
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

    # 6. Build and Assemble End-to-End Unified Model
    print("\n" + "=" * 60)
    print("ASSEMBLING UNIFIED END-TO-END VIDEO MODEL")
    print("=" * 60)
    video_input = layers.Input(shape=(None, 224, 224, 3), name="video_frames_input")
    x = layers.TimeDistributed(spatial_extractor, name="time_distributed_spatial")(video_input)
    final_output = temporal_model(x)

    end_to_end_model = models.Model(inputs=video_input, outputs=final_output, name="UnifiedVideoDeepfakeDetector_V2")
    end_to_end_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    end_to_end_model.summary()

    # Save candidate model
    checkpoint_v2 = models_dir / "best_model_v2.h5"
    end_to_end_model.save(str(checkpoint_v2))
    print(f"Saved candidate model to {checkpoint_v2}")

    # If accuracy > 80% and AUC > 0.85, promote to production
    if acc > 0.80 and auc > 0.85:
        import shutil
        prod_model = models_dir / "best_model.h5"
        shutil.copy2(checkpoint_v2, prod_model)
        print(f"Model promoted to active production checkpoint: {prod_model}")

        import json
        calib_file = PROJECT_ROOT / "models" / "calibration.json"
        with open(calib_file, "r", encoding="utf-8") as f:
            calib_data = json.load(f)

        calib_data["video"] = {
            "threshold": best_thresh,
            "f1_score": float(f1),
            "accuracy": float(acc),
            "roc_auc": float(auc),
            "uncertainty_lower": round(best_thresh - 0.08, 2),
            "uncertainty_upper": round(best_thresh + 0.08, 2),
            "checkpoint": "models/saved_models/video/best_model_v2.h5",
        }
        with open(calib_file, "w", encoding="utf-8") as f:
            json.dump(calib_data, f, indent=4)
        print(f"Calibration updated in {calib_file}")

    print("\nVideo Model Training & Validation Complete!")


if __name__ == "__main__":
    main()
