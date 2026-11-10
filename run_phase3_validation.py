import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import os
import io
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from scipy.fftpack import dctn
from scipy.ndimage import convolve
from scipy.stats import kurtosis
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import config
from predict import _load_model, get_threshold

os.environ["KERAS_HOME"] = config.KERAS_HOME
os.environ["MPLCONFIGDIR"] = config.TEMP_FOLDER

print("=" * 80)
print("PHASE 3: COMPREHENSIVE EXPERIMENTAL VALIDATION PIPELINE")
print("=" * 80)

# Load existing Stage 2 model
model = _load_model("image")
calib_threshold = get_threshold("image")
print(f"Loaded existing model: best_model_v2.h5 | Calibrated Threshold: {calib_threshold:.4f}")

# 1. Dataset loading
benchmark_dir = Path("dataset/high_res_benchmark")
manifest_path = benchmark_dir / "manifest.json"

with open(manifest_path, "r") as f:
    high_res_manifest = json.load(f)

# SRM High-Pass Kernels
SRM_F1 = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32) * 0.5
SRM_F2 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32) * 0.5
SRM_F3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) * 0.25
LAPLACIAN = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

def extract_frequency_features(pil_img):
    gray = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape
    ch, cw = min(h, 512), min(w, 512)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    crop = gray[y0:y0+ch, x0:x0+cw]
    
    # SRM filter residuals
    res_srm1 = convolve(crop, SRM_F1, mode="reflect")
    res_srm2 = convolve(crop, SRM_F2, mode="reflect")
    res_srm3 = convolve(crop, SRM_F3, mode="reflect")
    res_lap = convolve(crop, LAPLACIAN, mode="reflect")
    
    srm_energy = float(np.mean(np.abs(res_srm1) + np.abs(res_srm2) + np.abs(res_srm3)))
    lap_energy = float(np.mean(np.abs(res_lap)))
    lap_kurt = float(kurtosis(res_lap.ravel()))
    
    # 2D-DCT
    dct_coeffs = dctn(crop, norm="ortho")
    power_spectrum = np.abs(dct_coeffs) ** 2
    total_energy = float(np.sum(power_spectrum))
    
    u = np.linspace(0, 1, ch)[:, None]
    v = np.linspace(0, 1, cw)[None, :]
    radius = np.sqrt(u**2 + v**2)
    
    lf_mask = radius < 0.25
    mf_mask = (radius >= 0.25) & (radius < 0.6)
    hf_mask = radius >= 0.6
    
    lf_energy = float(np.sum(power_spectrum[lf_mask]))
    hf_energy = float(np.sum(power_spectrum[hf_mask]))
    hf_to_lf_ratio = (hf_energy / (lf_energy + 1e-8)) * 1000
    hf_pct = (hf_energy / (total_energy + 1e-8)) * 100
    
    return {
        "srm_energy": srm_energy,
        "lap_energy": lap_energy,
        "lap_kurtosis": lap_kurt,
        "hf_to_lf_ratio": hf_to_lf_ratio,
        "hf_pct": hf_pct
    }

def extract_multiscale_inputs(pil_img):
    pil_img = pil_img.convert("RGB")
    w, h = pil_img.size
    
    # Global letterboxed/padded to 224x224
    arr_global = np.array(pil_img, dtype=np.float32)
    padded_global = tf.image.resize_with_pad(arr_global, 224, 224, method="bilinear")
    norm_global = tf.keras.applications.mobilenet_v2.preprocess_input(padded_global).numpy()
    
    crops = {"global": norm_global}
    
    if w >= 224 and h >= 224:
        # Native Center crop
        cx, cy = w // 2, h // 2
        center = pil_img.crop((cx - 112, cy - 112, cx + 112, cy + 112))
        norm_center = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.convert_to_tensor(np.array(center, dtype=np.float32))
        ).numpy()
        crops["center"] = norm_center
        
        # Upper third (facial/portrait focal zone)
        uy = int(h * 0.35)
        upper = pil_img.crop((cx - 112, max(0, uy - 112), cx + 112, max(0, uy - 112) + 224))
        norm_upper = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.convert_to_tensor(np.array(upper, dtype=np.float32))
        ).numpy()
        crops["upper"] = norm_upper
        
    if w >= 512 and h >= 512:
        cx, cy = w // 2, h // 2
        mid = pil_img.crop((cx - 256, cy - 256, cx + 256, cy + 256)).resize((224, 224), Image.Resampling.BILINEAR)
        norm_mid = tf.keras.applications.mobilenet_v2.preprocess_input(
            tf.convert_to_tensor(np.array(mid, dtype=np.float32))
        ).numpy()
        crops["mid512"] = norm_mid
        
    return crops

# Prepare evaluation samples
eval_samples = []
for item in high_res_manifest:
    img_path = benchmark_dir / item["label"] / item["filename"]
    if img_path.exists():
        eval_samples.append({
            "path": img_path,
            "label_str": item["label"],
            "y_true": 1 if item["label"] == "real" else 0, # 1: Real, 0: Fake
            "category": "high_res",
            "name": item["filename"]
        })

# Also include CIFAKE samples for standard AI testing
cifake_fake_samples = list(Path("dataset/benchmark_cifake/test/fake").glob("*.png"))[:20]
cifake_real_samples = list(Path("dataset/benchmark_cifake/test/real").glob("*.png"))[:20]

for p in cifake_fake_samples:
    eval_samples.append({
        "path": p,
        "label_str": "fake",
        "y_true": 0,
        "category": "standard_ai_cifake",
        "name": p.name
    })

for p in cifake_real_samples:
    eval_samples.append({
        "path": p,
        "label_str": "real",
        "y_true": 1,
        "category": "standard_cifar_real",
        "name": p.name
    })

print(f"Total benchmark evaluation samples assembled: {len(eval_samples)}")

# Run inference across approaches
processed_data = []

for sample in eval_samples:
    with Image.open(sample["path"]) as img:
        im_rgb = img.convert("RGB")
        w, h = im_rgb.size
        crops = extract_multiscale_inputs(im_rgb)
        freq = extract_frequency_features(im_rgb)
        
    crop_preds = {}
    for c_name, c_tensor in crops.items():
        batch = np.expand_dims(c_tensor, axis=0)
        p_val = float(model.predict(batch, verbose=0)[0][0])
        crop_preds[c_name] = p_val
        
    p_global = crop_preds.get("global")
    p_center = crop_preds.get("center", p_global)
    p_upper = crop_preds.get("upper", p_global)
    p_mid512 = crop_preds.get("mid512", p_global)
    
    # Approach B: Multi-Scale Aggregation Strategies
    p_multiscale_mean = float(np.mean(list(crop_preds.values())))
    p_multiscale_min = float(np.min(list(crop_preds.values())))
    p_multiscale_weighted = float(0.5 * p_global + 0.3 * p_center + 0.2 * p_upper)
    
    # Approach C: Frequency Residual Score
    # Synthetic images typically have elevated SRM energy (>0.035) and lower kurtosis (<20)
    # Calibrated linear score map: P_freq_real is lower when SRM energy is high
    # Normalizing SRM energy roughly [0.01 -> Real(1.0), 0.06 -> Fake(0.0)]
    norm_srm = np.clip((freq["srm_energy"] - 0.010) / (0.055 - 0.010), 0.0, 1.0)
    p_freq_real = float(1.0 - norm_srm)
    
    # Approach D: Combined Multi-Scale + Frequency Fusion
    # 60% Multi-scale weighted + 40% Frequency residual
    p_combined = float(0.60 * p_multiscale_weighted + 0.40 * p_freq_real)
    
    processed_data.append({
        "name": sample["name"],
        "category": sample["category"],
        "y_true": sample["y_true"],
        "p_approach_a": p_global,
        "p_approach_b": p_multiscale_weighted,
        "p_approach_b_min": p_multiscale_min,
        "p_approach_c": p_freq_real,
        "p_approach_d": p_combined,
        "srm_energy": freq["srm_energy"],
        "lap_kurtosis": freq["lap_kurtosis"],
        "hf_to_lf_ratio": freq["hf_to_lf_ratio"]
    })

# Compute full metrics function
def compute_metrics(y_true, y_probs, thresh=0.52):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= thresh).astype(int) # 1: Real, 0: Fake
    
    acc = accuracy_score(y_true, y_pred)
    prec_real = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_real = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1_real = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    prec_fake = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    rec_fake = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1_fake = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.5
        
    # In binary classification where 0=Fake, 1=Real:
    # Confusion matrix:
    # row 0 (True Fake): [TN (Pred Fake), FP (Pred Real)]
    # row 1 (True Real): [FN (Pred Fake), TP (Pred Real)]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # FPR: Fake misclassified as Real = FP / (FP + TN)
    fpr_fake_as_real = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    # FNR: Real misclassified as Fake = FN / (FN + TP)
    fnr_real_as_fake = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "accuracy": acc,
        "roc_auc": auc,
        "prec_real": prec_real,
        "rec_real": rec_real,
        "f1_real": f1_real,
        "prec_fake": prec_fake,
        "rec_fake": rec_fake,
        "f1_fake": f1_fake,
        "fpr": fpr_fake_as_real,
        "fnr": fnr_real_as_fake,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }

print("\n" + "=" * 90)
print("1. EVALUATION ON HIGH-RESOLUTION SUBSET (Real Camera Photos vs High-Res AI Images)")
print("=" * 90)
hr_data = [d for d in processed_data if d["category"] == "high_res"]
y_hr = [d["y_true"] for d in hr_data]

approaches = [
    ("Approach A: Existing best_model_v2.h5 (Global Resize)", [d["p_approach_a"] for d in hr_data]),
    ("Approach B: Multi-Scale Preprocessing (Weighted Crops)", [d["p_approach_b"] for d in hr_data]),
    ("Approach B-Min: Multi-Scale Preprocessing (Max-Fake Crop)", [d["p_approach_b_min"] for d in hr_data]),
    ("Approach C: Frequency-Domain Feature Model (SRM Residuals)", [d["p_approach_c"] for d in hr_data]),
    ("Approach D: Combined Multi-Scale + Frequency Fusion", [d["p_approach_d"] for d in hr_data]),
]

print(f"{'Approach':<45} | {'Acc':<6} | {'AUC':<6} | {'Fake F1':<7} | {'Real F1':<7} | {'FPR (Fake->Real)':<16} | {'FNR (Real->Fake)'}")
print("-" * 115)
hr_results = {}
for name, probs in approaches:
    m = compute_metrics(y_hr, probs, thresh=calib_threshold)
    hr_results[name] = m
    print(f"{name:<45} | {m['accuracy']*100:5.1f}% | {m['roc_auc']:6.4f} | {m['f1_fake']:7.4f} | {m['f1_real']:7.4f} | {m['fpr']*100:15.1f}% | {m['fnr']*100:15.1f}%")

print("\n" + "=" * 90)
print("2. EVALUATION ON OVERALL COMPOSITE BENCHMARK (High-Res + CIFAKE Standard AI)")
print("=" * 90)
y_all = [d["y_true"] for d in processed_data]
all_results = {}
for name, _ in approaches:
    key = "p_approach_a" if "Approach A" in name else (
        "p_approach_b" if "Weighted" in name else (
            "p_approach_b_min" if "Max-Fake" in name else (
                "p_approach_c" if "Approach C" in name else "p_approach_d"
            )
        )
    )
    probs = [d[key] for d in processed_data]
    m = compute_metrics(y_all, probs, thresh=calib_threshold)
    all_results[name] = m
    print(f"{name:<45} | {m['accuracy']*100:5.1f}% | {m['roc_auc']:6.4f} | {m['f1_fake']:7.4f} | {m['f1_real']:7.4f} | {m['fpr']*100:15.1f}% | {m['fnr']*100:15.1f}%")

print("\n" + "=" * 90)
print("3. COMPRESSION STABILITY EXPERIMENT (JPEG Quality Q=100 down to Q=15)")
print("=" * 90)
ai_img_path = Path("../ai-generated-8622236_1280.jpg")
real_img_path = Path("static/uploads/cf36c9647ce14667b790c93c3595fca9_IMG_20240207_174754.jpg")

compression_results = []
if ai_img_path.exists() and real_img_path.exists():
    with Image.open(ai_img_path) as ai_im, Image.open(real_img_path) as real_im:
        for q in [100, 85, 70, 50, 30, 15]:
            buf_ai = io.BytesIO()
            ai_im.save(buf_ai, format="JPEG", quality=q)
            buf_ai.seek(0)
            comp_ai = Image.open(buf_ai)
            
            buf_real = io.BytesIO()
            real_im.save(buf_real, format="JPEG", quality=q)
            buf_real.seek(0)
            comp_real = Image.open(buf_real)
            
            # Predict for AI
            crops_ai = extract_multiscale_inputs(comp_ai)
            freq_ai = extract_frequency_features(comp_ai)
            p_ai_a = float(model.predict(np.expand_dims(crops_ai["global"], axis=0), verbose=0)[0][0])
            p_ai_b = float(0.5 * p_ai_a + 0.3 * float(model.predict(np.expand_dims(crops_ai.get("center", crops_ai["global"]), axis=0), verbose=0)[0][0]) + 0.2 * float(model.predict(np.expand_dims(crops_ai.get("upper", crops_ai["global"]), axis=0), verbose=0)[0][0]))
            norm_srm_ai = np.clip((freq_ai["srm_energy"] - 0.010) / (0.055 - 0.010), 0.0, 1.0)
            p_ai_c = float(1.0 - norm_srm_ai)
            p_ai_d = float(0.6 * p_ai_b + 0.4 * p_ai_c)
            
            # Predict for Real
            crops_real = extract_multiscale_inputs(comp_real)
            freq_real = extract_frequency_features(comp_real)
            p_real_a = float(model.predict(np.expand_dims(crops_real["global"], axis=0), verbose=0)[0][0])
            p_real_b = float(0.5 * p_real_a + 0.3 * float(model.predict(np.expand_dims(crops_real.get("center", crops_real["global"]), axis=0), verbose=0)[0][0]) + 0.2 * float(model.predict(np.expand_dims(crops_real.get("upper", crops_real["global"]), axis=0), verbose=0)[0][0]))
            norm_srm_real = np.clip((freq_real["srm_energy"] - 0.010) / (0.055 - 0.010), 0.0, 1.0)
            p_real_c = float(1.0 - norm_srm_real)
            p_real_d = float(0.6 * p_real_b + 0.4 * p_real_c)
            
            compression_results.append({
                "quality": q,
                "ai_p_app_a": p_ai_a,
                "ai_p_app_b": p_ai_b,
                "ai_p_app_c": p_ai_c,
                "ai_p_app_d": p_ai_d,
                "real_p_app_a": p_real_a,
                "real_p_app_b": p_real_b,
                "real_p_app_c": p_real_c,
                "real_p_app_d": p_real_d,
                "ai_srm": freq_ai["srm_energy"],
                "real_srm": freq_real["srm_energy"],
            })
            print(f"JPEG Q={q:<3} | AI Fake: App A={p_ai_a:.4f}, App B={p_ai_b:.4f}, App C={p_ai_c:.4f}, App D={p_ai_d:.4f} | Real Photo: App A={p_real_a:.4f}, App B={p_real_b:.4f}, App D={p_real_d:.4f}")

# Save results json
results_output = {
    "hr_results": hr_results,
    "all_results": all_results,
    "compression_results": compression_results,
    "individual_data": processed_data
}

output_path = Path("reports/phase3_experimental_validation_results.json")
with open(output_path, "w") as f:
    json.dump(results_output, f, indent=4)

print(f"\n[OK] Validation results saved to {output_path.resolve()}")
