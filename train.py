import argparse
import os
import tensorflow as tf
from pathlib import Path

from preprocessing.clean_image import get_image_dataset
from preprocessing.clean_video import get_video_dataset
from preprocessing.clean_audio import get_audio_dataset
from models.image_model import build_image_model
from models.video_model import build_video_model
from models.audio_model import build_audio_model

SUPPORTED_DATASET_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
    "video": {".mp4", ".avi", ".mov", ".mkv", ".webm"},
    "audio": {".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a"},
}


def get_hardware_strategy():
    """Detect hardware and set memory growth for GPUs."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s). Using GPU.")
        except RuntimeError as e:
            print(e)
    else:
        print("No GPU found. Training on CPU. This will be slow.")


def get_class_weights(dataset_dir, modality):
    extensions = SUPPORTED_DATASET_EXTENSIONS[modality]
    counts = {
        class_id: sum(
            1
            for path in (Path(dataset_dir) / class_name).rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        for class_id, class_name in enumerate(("fake", "real"))
    }
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Both fake and real training samples are required for {modality}.")

    total = sum(counts.values())
    return {
        class_id: total / (2 * count)
        for class_id, count in counts.items()
    }


def train_modality(modality, batch_size=32, epochs=10, resume=False, checkpoint_path=None):
    get_hardware_strategy()
    
    project_root = Path(__file__).parent.resolve()
    dataset_dir = project_root / "dataset" / modality
    models_dir = project_root / "models" / "saved_models" / modality
    logs_dir = project_root / "logs" / modality
    
    models_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    if checkpoint_path is None:
        checkpoint_path = models_dir / "best_model.h5"
        training_log_path = logs_dir / "training_log.csv"
    else:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        training_log_path = checkpoint_path.parent / "training_log.csv"
    
    # Check if dataset directories exist
    if not (dataset_dir / "train").exists() or not (dataset_dir / "validation").exists():
        print(f"Error: {dataset_dir / 'train'} or {dataset_dir / 'validation'} not found.")
        print("Please ensure dataset preparation has populated these folders.")
        return

    print(f"=== Starting Training for {modality.upper()} ===")
    
    if modality == 'image':
        train_ds = get_image_dataset(dataset_dir / "train", batch_size=batch_size, subset="train")
        val_ds = get_image_dataset(dataset_dir / "validation", batch_size=batch_size, subset="validation", shuffle=False)
        model = build_image_model()
    elif modality == 'video':
        # Smaller batch size for video to fit in 6GB VRAM
        video_batch = min(batch_size, 4)
        train_ds = get_video_dataset(dataset_dir / "train", batch_size=video_batch)
        val_ds = get_video_dataset(dataset_dir / "validation", batch_size=video_batch, shuffle=False)
        model = build_video_model()
    elif modality == 'audio':
        train_ds = get_audio_dataset(dataset_dir / "train", batch_size=batch_size)
        val_ds = get_audio_dataset(dataset_dir / "validation", batch_size=batch_size, shuffle=False)
        model = build_audio_model()
    else:
        raise ValueError("Invalid modality")

    if resume and checkpoint_path.exists():
        print(f"Resuming from checkpoint: {checkpoint_path}")
        model.load_weights(str(checkpoint_path))
        
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            save_best_only=True,
            monitor='val_loss',
            mode='min',
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(str(training_log_path), append=resume),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=get_class_weights(dataset_dir / "train", modality),
    )

    if checkpoint_path.resolve() == (models_dir / "best_model.h5").resolve():
        from calibrate_thresholds import calibrate_modality, save_calibration

        calibration = calibrate_modality(modality)
        if calibration is None:
            raise RuntimeError(f"Could not calibrate the trained {modality} model.")
        save_calibration({modality: calibration})

    print(f"=== Training Complete for {modality.upper()} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Deepfake Models")
    parser.add_argument("--modality", choices=["image", "video", "audio"], required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument(
        "--checkpoint-path",
        help="Optional output path for a candidate checkpoint; defaults to the active model path",
    )
    args = parser.parse_args()
    
    train_modality(
        args.modality,
        args.batch_size,
        args.epochs,
        args.resume,
        checkpoint_path=args.checkpoint_path,
    )
