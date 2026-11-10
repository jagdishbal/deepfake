import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# config MUST be imported before tensorflow to set KERAS_HOME, TEMP, etc.
import config
import os
import random
import numpy as np
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
from preprocessing.clean_image import preprocess_image_tensor
from models.image_model import build_image_model

MODELS_DIR = PROJECT_ROOT / "models" / "saved_models" / "image"
DATASET_DIR = PROJECT_ROOT / "dataset" / "image"
LOGS_DIR = PROJECT_ROOT / "logs" / "image"

INITIAL_WEIGHTS = MODELS_DIR / "best_model.h5"
CANDIDATE_CHECKPOINT = MODELS_DIR / "candidate_model_v2.h5"
LOG_PATH = LOGS_DIR / "training_candidate_log.csv"

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

def build_augmented_dataset(directory, batch_size=32, is_train=True):
    dir_path = Path(directory)
    dataset = tf.keras.utils.image_dataset_from_directory(
        str(dir_path),
        labels='inferred',
        label_mode='binary',
        class_names=['fake', 'real'],
        batch_size=batch_size,
        image_size=(224, 224),
        crop_to_aspect_ratio=True,
        shuffle=is_train,
        seed=42 if is_train else None,
    )
    
    if is_train:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomContrast(0.12),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomZoom(0.05),
        ])
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        
    dataset = dataset.map(
        lambda x, y: (
            tf.keras.applications.mobilenet_v2.preprocess_input(x),
            y,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

def train_candidate():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    print("=== Preparing Data for Image Candidate Retraining ===")
    train_ds = build_augmented_dataset(DATASET_DIR / "train", batch_size=32, is_train=True)
    val_ds = build_augmented_dataset(DATASET_DIR / "validation", batch_size=32, is_train=False)
    class_weights = get_class_weights(DATASET_DIR / "train")
    
    print("\n" + "=" * 55)
    print("STAGE 1: Pretrained Head Training (Frozen Backbone)")
    print("=" * 55)
    model = build_image_model(weights=None, learning_rate=1e-4, fine_tune_at=None)
    if INITIAL_WEIGHTS.exists():
        print(f"Loading initial weights from: {INITIAL_WEIGHTS}")
        model.load_weights(str(INITIAL_WEIGHTS))
    else:
        print("Initial weights not found; loading ImageNet weights")
        model = build_image_model(weights="imagenet", learning_rate=1e-4, fine_tune_at=None)

    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=3, restore_best_weights=True)
    ]
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=5,
        callbacks=callbacks_stage1,
        class_weight=class_weights,
    )
    
    print("\n" + "=" * 55)
    print("STAGE 2: Fine-Tuning Backbone (Unfrozen from Layer 115)")
    print("=" * 55)
    base_model = model.layers[0]
    base_model.trainable = True
    for layer in base_model.layers[:115]:
        layer.trainable = False
        
    trainable_count = sum(np.prod(p.shape) for p in model.trainable_weights)
    print(f"Trainable parameters: {trainable_count:,}")
    
    loss_fn = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.04)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1.5e-5),
        loss=loss_fn,
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    
    callbacks_stage2 = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(CANDIDATE_CHECKPOINT),
            save_best_only=True,
            monitor="val_auc",
            mode="max",
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(LOG_PATH), append=False),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
    ]
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        callbacks=callbacks_stage2,
        class_weight=class_weights,
    )
    
    print(f"\nCandidate training complete. Model saved to: {CANDIDATE_CHECKPOINT}")

if __name__ == "__main__":
    train_candidate()
