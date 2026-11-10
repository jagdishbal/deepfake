import config
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import io

from predict import _load_model, get_threshold, _preprocess_image

model = _load_model("image")
threshold = get_threshold("image")

print("=" * 70)
print("EXPERIMENT 1: RESOLUTION SWEEP (32x32, 128x128, 256x256, 512x512, Native)")
print("=" * 70)

test_cases = [
    ("High-Res AI Portrait", Path("../ai-generated-8622236_1280.jpg")),
    ("Compressed Web AI Image", Path("../OIP.jpg")),
    ("Genuine High-Res Camera Photo", Path("static/uploads/cf36c9647ce14667b790c93c3595fca9_IMG_20240207_174754.jpg")),
    ("Genuine Portrait Photo", Path("static/uploads/4d3090911ad442b0ada6c6e82ab9e30c_bicky.jpg")),
]

# Add a 32x32 CIFAKE image
cifake_fake = list(Path("dataset/benchmark_cifake/test/fake").glob("*.png"))[0]
cifake_real = list(Path("dataset/benchmark_cifake/test/real").glob("*.png"))[0]
test_cases.append(("CIFAKE 32x32 Synthetic (Stable Diff)", cifake_fake))
test_cases.append(("CIFAKE 32x32 Real (CIFAR-10)", cifake_real))

resolutions = [(32, 32), (128, 128), (256, 256), (512, 512), "native"]

def run_prediction_on_pil(pil_img):
    # Convert to RGB
    pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
    # Pad and resize to 224x224
    padded = tf.image.resize_with_pad(tensor, 224, 224, method="bilinear")
    normed = tf.keras.applications.mobilenet_v2.preprocess_input(padded)
    batch = tf.expand_dims(normed, axis=0)
    prob_real = float(model.predict(batch, verbose=0)[0][0])
    pred = "Real" if prob_real >= threshold else "Fake"
    return pred, prob_real

print(f"{'Image Name':<35} | {'Native Res':<12} | {'Tested Res':<10} | {'Pred':<5} | {'Prob(Real)':<10}")
print("-" * 80)

for label, p in test_cases:
    if not p.exists():
        continue
    with Image.open(p) as img:
        orig_res = f"{img.width}x{img.height}"
        for res in resolutions:
            if res == "native":
                test_img = img.copy()
                res_str = "Native"
            else:
                test_img = img.resize(res, Image.Resampling.BILINEAR)
                res_str = f"{res[0]}x{res[1]}"
            
            pred, prob = run_prediction_on_pil(test_img)
            print(f"{label:<35} | {orig_res:<12} | {res_str:<10} | {pred:<5} | {prob:.4f}")
    print("-" * 80)

print("\n" + "=" * 70)
print("EXPERIMENT 2: COMPRESSION & INTERPOLATION EFFECTS ON HIGH-RES AI IMAGE")
print("=" * 70)
ai_img_path = Path("../ai-generated-8622236_1280.jpg")
if ai_img_path.exists():
    with Image.open(ai_img_path) as orig_ai:
        print("A. JPEG Quality Levels on High-Res AI Portrait:")
        for q in [100, 85, 70, 50, 30, 15]:
            buf = io.BytesIO()
            orig_ai.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            with Image.open(buf) as comp_img:
                pred, prob = run_prediction_on_pil(comp_img)
                print(f"  JPEG Quality {q:<3} -> Pred: {pred:<5} | Prob(Real): {prob:.4f}")

        print("\nB. Resampling Algorithms (Resized to 224x224):")
        for name, algo in [("Bilinear", Image.Resampling.BILINEAR),
                           ("Bicubic", Image.Resampling.BICUBIC),
                           ("Nearest", Image.Resampling.NEAREST),
                           ("Lanczos", Image.Resampling.LANCZOS)]:
            resampled = orig_ai.resize((224, 224), algo)
            pred, prob = run_prediction_on_pil(resampled)
            print(f"  Resampling {name:<8} -> Pred: {pred:<5} | Prob(Real): {prob:.4f}")

print("\n" + "=" * 70)
print("EXPERIMENT 3: CROPPING TO FACE / HIGH-FREQUENCY DETAILS")
print("=" * 70)
if ai_img_path.exists():
    with Image.open(ai_img_path) as orig_ai:
        w, h = orig_ai.size
        # Center crop (eyes / nose / skin texture)
        center_crop = orig_ai.crop((w//4, h//4, 3*w//4, 3*h//4))
        pred, prob = run_prediction_on_pil(center_crop)
        print(f"Center Crop (50% area)       -> Pred: {pred:<5} | Prob(Real): {prob:.4f}")
        
        # 224x224 direct crop of facial skin / eyes
        face_crop = orig_ai.crop((w//2 - 112, h//3 - 112, w//2 + 112, h//3 + 112))
        pred, prob = run_prediction_on_pil(face_crop)
        print(f"Direct 224x224 Face Crop     -> Pred: {pred:<5} | Prob(Real): {prob:.4f}")

        # Direct 224x224 crop of background
        bg_crop = orig_ai.crop((0, 0, 224, 224))
        pred, prob = run_prediction_on_pil(bg_crop)
        print(f"Direct 224x224 Background Crop -> Pred: {pred:<5} | Prob(Real): {prob:.4f}")
