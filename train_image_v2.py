import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config # Sets KERAS_HOME, TEMP, etc. before importing tensorflow
import os
import random
import numpy as np
import tensorflow as tf

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

from preprocessing.clean_image import get_image_dataset
from models.image_model import build_image_model

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models" / "saved_models" / "image"
LOGS_DIR = PROJECT_ROOT / "logs" / "image"
DATASET_DIR = PROJECT_ROOT / "dataset" / "image"

EXISTING_CHECKPOINT = MODELS_DIR / "best_model.h5"
NEW_CHECKPOINT = MODELS_DIR / "best_model_v2.h5"
LOG_PATH = LOGS_DIR / "training_log_v2.csv"

def get_class_weights(dataset_dir):
    fake_count = len(list((dataset_dir / "fake").glob("*")))
    real_count = len(list((dataset_dir / "real").glob("*")))
    total = fake_count + real_count
    weights = {
        0: total / (2.0 * fake_count),
        1: total / (2.0 * real_count),
    }
    print(f"Dataset Counts -> Fake (0): {fake_count}, Real (1): {real_count}")
    print(f"Computed Class Weights -> Fake: {weights[0]:.4f}, Real: {weights[1]:.4f}")
    return weights

def train_image_v2(batch_size=32, stage1_epochs=5, stage2_epochs=6):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading Data for Image V2 Retraining ===")
    train_ds = get_image_dataset(DATASET_DIR / "train", batch_size=batch_size, subset="train")
    val_ds = get_image_dataset(DATASET_DIR / "validation", batch_size=batch_size, subset="validation", shuffle=False)
    class_weights = get_class_weights(DATASET_DIR / "train")

    print("\n" + "="*50)
    print("STAGE 1: Loading Pretrained Model & Training Classification Head")
    print("="*50)
    model = build_image_model(weights=None, learning_rate=1e-4, fine_tune_at=None)
    
    if EXISTING_CHECKPOINT.exists():
        print(f"Loading pretrained weights from: {EXISTING_CHECKPOINT}")
        model.load_weights(str(EXISTING_CHECKPOINT))
    else:
        print("Warning: Existing checkpoint not found, starting from random initialization.")

    callbacks_stage1 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(NEW_CHECKPOINT),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(LOG_PATH), append=False),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        callbacks=callbacks_stage1,
        class_weight=class_weights,
    )

    print("\n" + "="*50)
    print("STAGE 2: Fine-Tuning Top MobileNetV2 Blocks (Learning Rate: 1e-5)")
    print("="*50)

    # Unfreeze from layer 120 onwards
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:120]:
        layer.trainable = False

    trainable_count = sum(np.prod(p.shape) for p in model.trainable_weights)
    print(f"Backbone unfrozen from layer 120. Trainable parameters: {trainable_count:,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )

    callbacks_stage2 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(NEW_CHECKPOINT),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(LOG_PATH), append=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    ]

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        callbacks=callbacks_stage2,
        class_weight=class_weights,
    )

    print(f"\nTraining Complete! New model checkpoint saved at: {NEW_CHECKPOINT}")
    print(f"Original model checkpoint remains untouched at: {EXISTING_CHECKPOINT}")
    return model

if __name__ == "__main__":
    train_image_v2()
