import config
import json
from pathlib import Path
import tensorflow as tf
import numpy as np

from predict import _load_model, get_threshold, _preprocess_image, predict_file

print("=" * 60)
print("1. VERIFYING MODEL CHECKPOINT EXISTENCE AND INTEGRITY")
print("=" * 60)

v1_path = Path("models/saved_models/image/best_model.h5")
v2_path = Path("models/saved_models/image/best_model_v2.h5")
backup_v1 = Path("models/backups/image_best_model.h5")
calib_path = Path("models/calibration.json")

print(f"Original Checkpoint (v1): {v1_path} -> Exists: {v1_path.exists()}, Size: {v1_path.stat().st_size:,} bytes")
print(f"Original Backup (v1):     {backup_v1} -> Exists: {backup_v1.exists()}, Size: {backup_v1.stat().st_size:,} bytes")
print(f"New Checkpoint (v2):      {v2_path} -> Exists: {v2_path.exists()}, Size: {v2_path.stat().st_size:,} bytes")
print(f"Calibration File:         {calib_path} -> Exists: {calib_path.exists()}, Size: {calib_path.stat().st_size:,} bytes")

print("\n" + "=" * 60)
print("2. VERIFYING ARCHITECTURE & LOADING VIA PREDICT PIPELINE")
print("=" * 60)

model = _load_model("image")
print(f"Loaded Model Class: {type(model)}")
print(f"Model Input Shape:  {model.input_shape}")
print(f"Model Output Shape: {model.output_shape}")
print(f"Number of Layers:   {len(model.layers)}")
for idx, layer in enumerate(model.layers):
    print(f"  Layer {idx}: {layer.name} ({type(layer).__name__}), trainable={layer.trainable}")

print("\n" + "=" * 60)
print("3. VERIFYING THRESHOLD AND CALIBRATION")
print("=" * 60)
threshold = get_threshold("image")
print(f"Configured Threshold for 'image': {threshold}")
with open(calib_path, "r") as f:
    calib = json.load(f)
print("Full Image Calibration Block:")
print(json.dumps(calib.get("image", {}), indent=4))

print("\n" + "=" * 60)
print("4. VERIFYING INPUT PREPROCESSING")
print("=" * 60)
dummy_img_path = Path("../OIP.jpg")
if dummy_img_path.exists():
    processed_tensor = _preprocess_image(dummy_img_path)
    print(f"Processed shape: {processed_tensor.shape}")
    print(f"Min value: {processed_tensor.min():.4f}, Max value: {processed_tensor.max():.4f}")
    print(f"Expected range: approx [-1.0, 1.0] (MobileNetV2 normalized)")
    assert processed_tensor.shape == (1, 224, 224, 3), f"Incorrect shape {processed_tensor.shape}"
    assert -1.05 <= processed_tensor.min() <= 0.0, f"Unexpected min {processed_tensor.min()}"
    assert 0.0 <= processed_tensor.max() <= 1.05, f"Unexpected max {processed_tensor.max()}"
    print("Preprocessing verification PASSED.")
else:
    print("Dummy image not found for tensor range check.")
