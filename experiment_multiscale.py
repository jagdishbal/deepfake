import config
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

from predict import _load_model, get_threshold

model = _load_model("image")
threshold = get_threshold("image")

print("=" * 80)
print("STEP 3: MULTI-SCALE PREPROCESSING & PATCH SAMPLING EXPERIMENT")
print("=" * 80)

def extract_multiscale_patches(pil_img):
    """
    Extracts global image view + native 224x224 crops across key spatial regions.
    """
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    
    # 1. Global View (Padded symmetrically)
    arr_global = np.array(pil_img)
    tensor_global = tf.convert_to_tensor(arr_global, dtype=tf.float32)
    padded_global = tf.image.resize_with_pad(tensor_global, 224, 224, method="bilinear")
    norm_global = tf.keras.applications.mobilenet_v2.preprocess_input(padded_global)
    
    patches = {
        "global": norm_global.numpy()
    }
    
    crop_size = 224
    if w >= crop_size and h >= crop_size:
        # 2. Native Center Crop (Face / subject center)
        cx, cy = w // 2, h // 2
        center = pil_img.crop((cx - 112, cy - 112, cx + 112, cy + 112))
        patches["center_native"] = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.convert_to_tensor(np.array(center), dtype=tf.float32)
        ).numpy()
        
        # 3. Upper-Third Portrait Crop (Eyes / forehead)
        uy = int(h * 0.35)
        upper = pil_img.crop((cx - 112, max(0, uy - 112), cx + 112, max(0, uy - 112) + 224))
        patches["upper_third"] = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.convert_to_tensor(np.array(upper), dtype=tf.float32)
        ).numpy()
        
        # 4. Corner Detail Crops (Hair/boundary/background blend)
        pad_off = min(w // 8, h // 8, 100)
        tl = pil_img.crop((pad_off, pad_off, pad_off + 224, pad_off + 224))
        patches["top_left"] = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.convert_to_tensor(np.array(tl), dtype=tf.float32)
        ).numpy()
        
        br = pil_img.crop((w - pad_off - 224, h - pad_off - 224, w - pad_off, h - pad_off))
        patches["bottom_right"] = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.convert_to_tensor(np.array(br), dtype=tf.float32)
        ).numpy()
        
    # 5. Intermediate 512x512 Crop Downscaled to 224x224 (Mid-frequency scale)
    if w >= 512 and h >= 512:
        cx, cy = w // 2, h // 2
        mid = pil_img.crop((cx - 256, cy - 256, cx + 256, cy + 256)).resize((224, 224), Image.Resampling.BILINEAR)
        patches["mid_scale_512"] = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.convert_to_tensor(np.array(mid), dtype=tf.float32)
        ).numpy()
        
    return patches

benchmark_dir = Path("dataset/high_res_benchmark")
manifest_file = benchmark_dir / "manifest.json"

with open(manifest_file, "r") as f:
    benchmark_items = json.load(f)

# Group evaluation results
results = []

print(f"{'Filename':<18} | {'True':<4} | {'Global P':<8} | {'Center':<8} | {'Upper':<8} | {'Mid512':<8} | {'Mean P':<8} | {'Min P':<8} | {'Best Strategy'}")
print("-" * 110)

for item in benchmark_items:
    img_path = benchmark_dir / item["label"] / item["filename"]
    if not img_path.exists():
        continue
        
    with Image.open(img_path) as img:
        patches = extract_multiscale_patches(img)
        
    patch_scores = {}
    for p_name, p_tensor in patches.items():
        batch = np.expand_dims(p_tensor, axis=0)
        p_score = float(model.predict(batch, verbose=0)[0][0])
        patch_scores[p_name] = p_score
        
    p_global = patch_scores.get("global", 0.5)
    p_center = patch_scores.get("center_native", p_global)
    p_upper = patch_scores.get("upper_third", p_global)
    p_mid512 = patch_scores.get("mid_scale_512", p_global)
    
    all_scores = list(patch_scores.values())
    p_mean = float(np.mean(all_scores))
    p_min = float(np.min(all_scores))
    p_weighted = float(0.5 * p_global + 0.5 * p_center)
    
    y_true = item["label"] # "real" or "fake"
    
    # Classify under each strategy
    pred_global = "real" if p_global >= threshold else "fake"
    pred_mean = "real" if p_mean >= threshold else "fake"
    pred_min = "real" if p_min >= threshold else "fake"
    pred_weighted = "real" if p_weighted >= threshold else "fake"
    
    results.append({
        "item": item["filename"],
        "true": y_true,
        "global": p_global,
        "center": p_center,
        "upper": p_upper,
        "mid512": p_mid512,
        "mean": p_mean,
        "min": p_min,
        "weighted": p_weighted,
        "pred_global": pred_global,
        "pred_mean": pred_mean,
        "pred_min": pred_min,
        "pred_weighted": pred_weighted,
    })
    
    center_str = f"{p_center:.4f}" if "center_native" in patch_scores else "N/A"
    upper_str = f"{p_upper:.4f}" if "upper_third" in patch_scores else "N/A"
    mid_str = f"{p_mid512:.4f}" if "mid_scale_512" in patch_scores else "N/A"
    
    # Determine best matching strategy
    correct_strats = []
    if pred_global == y_true: correct_strats.append("Global")
    if pred_mean == y_true: correct_strats.append("Mean")
    if pred_min == y_true: correct_strats.append("Min")
    if pred_weighted == y_true: correct_strats.append("Weighted")
    strat_text = ", ".join(correct_strats) if correct_strats else "None"
    
    print(f"{item['filename']:<18} | {y_true:<4} | {p_global:<8.4f} | {center_str:<8} | {upper_str:<8} | {mid_str:<8} | {p_mean:<8.4f} | {p_min:<8.4f} | {strat_text}")

print("-" * 110)

# Summary performance by strategy
print("\n" + "=" * 80)
print("AGGREGATION STRATEGY ACCURACY COMPARISON ON HIGH-RES BENCHMARK:")
print("=" * 80)
strategies = ["global", "mean", "min", "weighted"]
for strat in strategies:
    correct_real = sum(1 for r in results if r["true"] == "real" and r[f"pred_{strat}"] == "real")
    total_real = sum(1 for r in results if r["true"] == "real")
    correct_fake = sum(1 for r in results if r["true"] == "fake" and r[f"pred_{strat}"] == "fake")
    total_fake = sum(1 for r in results if r["true"] == "fake")
    total = len(results)
    acc = (correct_real + correct_fake) / total * 100
    print(f"Strategy: {strat.capitalize():<10} | Real Sensitivity: {correct_real}/{total_real} ({correct_real/total_real*100:5.1f}%) | Fake Specificity: {correct_fake}/{total_fake} ({correct_fake/total_fake*100:5.1f}%) | Overall Acc: {acc:5.1f}%")
