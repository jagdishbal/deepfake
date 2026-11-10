import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score

from preprocessing.clean_image import get_image_dataset
from preprocessing.clean_video import get_video_dataset
from preprocessing.clean_audio import get_audio_dataset
from models.image_model import build_image_model
from models.video_model import build_video_model
from models.audio_model import build_audio_model


PROJECT_ROOT = Path(__file__).parent.resolve()
CALIBRATION_PATH = PROJECT_ROOT / "models" / "calibration.json"


def get_predictions(modality, model, ds):
    y_true = []
    y_probs = []
    for x, y in ds:
        probs = model.predict(x, verbose=0)
        y_true.extend(y.numpy().flatten())
        y_probs.extend(probs.flatten())
    return np.array(y_true), np.array(y_probs)

def calibrate_modality(modality):
    print(f"Calibrating {modality}...")
    val_dir = PROJECT_ROOT / "dataset" / modality / "validation"
    models_dir = PROJECT_ROOT / "models" / "saved_models" / modality
    checkpoint_path = models_dir / "best_model.h5"

    if not checkpoint_path.exists():
        print(f"No checkpoint found for {modality} at {checkpoint_path}")
        return None

    if modality == 'image':
        val_ds = get_image_dataset(val_dir, batch_size=32, subset="validation", shuffle=False)
        model = build_image_model()
    elif modality == 'video':
        val_ds = get_video_dataset(val_dir, batch_size=4, shuffle=False)
        model = build_video_model()
    elif modality == 'audio':
        val_ds = get_audio_dataset(val_dir, batch_size=32, shuffle=False)
        model = build_audio_model()

    model.load_weights(str(checkpoint_path))
    y_true, y_probs = get_predictions(modality, model, val_ds)

    best_threshold = 0.5
    best_f1 = -1.0

    if len(y_true) > 0:
        for threshold in np.arange(0.01, 1.00, 0.01):
            y_pred = (y_probs >= threshold).astype(int)
            macro_f1 = f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
            if macro_f1 > best_f1 or (
                np.isclose(macro_f1, best_f1)
                and abs(threshold - 0.5) < abs(best_threshold - 0.5)
            ):
                best_f1 = macro_f1
                best_threshold = float(threshold)

    print(
        f"Best threshold for {modality}: {best_threshold:.2f} "
        f"with validation macro F1: {best_f1:.4f}"
    )
    # Derive an uncertainty bound around the threshold
    return {
        "threshold": float(best_threshold),
        "f1_score": float(best_f1),
        "uncertainty_lower": max(0.0, float(best_threshold - 0.05)),
        "uncertainty_upper": min(1.0, float(best_threshold + 0.05))
    }


def save_calibration(results, output_path=CALIBRATION_PATH):
    output_path = Path(output_path)
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as calibration_file:
            calibration = json.load(calibration_file)
        if not isinstance(calibration, dict):
            raise ValueError(f"Calibration file must contain a JSON object: {output_path}")
    else:
        calibration = {}

    calibration.update(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as calibration_file:
        json.dump(calibration, calibration_file, indent=4)


if __name__ == '__main__':
    calibration = {}
    for mod in ['image', 'video', 'audio']:
        res = calibrate_modality(mod)
        if res:
            calibration[mod] = res
    
    save_calibration(calibration)
    print("Calibration complete.")
