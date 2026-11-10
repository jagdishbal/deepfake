import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.fftpack import dctn
from scipy.ndimage import convolve
from scipy.stats import kurtosis

print("=" * 80)
print("STEP 4: FREQUENCY-DOMAIN & HIGH-PASS RESIDUAL EXPERIMENT")
print("=" * 80)

# SRM High-Pass Kernels
SRM_F1 = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32) * 0.5
SRM_F2 = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32) * 0.5
SRM_F3 = np.array([[-1, 2, -1], [2, -4, 2], [-1, 2, -1]], dtype=np.float32) * 0.25
LAPLACIAN = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

def analyze_frequency_features(pil_img):
    # Convert to grayscale luminance
    gray = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0
    h, w = gray.shape
    
    # Crop center 512x512 for standardized spectral comparison
    ch, cw = min(h, 512), min(w, 512)
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    crop = gray[y0:y0+ch, x0:x0+cw]
    
    # 1. Spatial High-Pass Residuals
    res_srm1 = convolve(crop, SRM_F1, mode="reflect")
    res_srm2 = convolve(crop, SRM_F2, mode="reflect")
    res_srm3 = convolve(crop, SRM_F3, mode="reflect")
    res_lap = convolve(crop, LAPLACIAN, mode="reflect")
    
    srm_energy = float(np.mean(np.abs(res_srm1) + np.abs(res_srm2) + np.abs(res_srm3)))
    lap_energy = float(np.mean(np.abs(res_lap)))
    lap_kurtosis = float(kurtosis(res_lap.ravel()))
    
    # 2. 2D-DCT Spectral Analysis
    dct_coeffs = dctn(crop, norm="ortho")
    power_spectrum = np.abs(dct_coeffs) ** 2
    total_energy = float(np.sum(power_spectrum))
    
    # Frequency radial partition
    u = np.linspace(0, 1, ch)[:, None]
    v = np.linspace(0, 1, cw)[None, :]
    radius = np.sqrt(u**2 + v**2)
    
    lf_mask = radius < 0.25
    mf_mask = (radius >= 0.25) & (radius < 0.6)
    hf_mask = radius >= 0.6
    
    lf_energy = float(np.sum(power_spectrum[lf_mask]))
    mf_energy = float(np.sum(power_spectrum[mf_mask]))
    hf_energy = float(np.sum(power_spectrum[hf_mask]))
    
    hf_to_lf_ratio = (hf_energy / (lf_energy + 1e-8)) * 1000
    hf_pct = (hf_energy / (total_energy + 1e-8)) * 100
    spectral_kurtosis = float(kurtosis(np.log1p(power_spectrum.ravel())))
    
    return {
        "srm_energy": srm_energy,
        "lap_energy": lap_energy,
        "lap_kurtosis": lap_kurtosis,
        "hf_to_lf_ratio": hf_to_lf_ratio,
        "hf_energy_pct": hf_pct,
        "spectral_kurtosis": spectral_kurtosis,
    }

benchmark_dir = Path("dataset/high_res_benchmark")
manifest_file = benchmark_dir / "manifest.json"

with open(manifest_file, "r") as f:
    benchmark_items = json.load(f)

# Also add OIP.jpg and CIFAKE samples
extra_samples = [
    {"filename": "OIP.jpg", "label": "fake", "path": Path("../OIP.jpg")},
    {"filename": "cifake_fake_000510.png", "label": "fake", "path": Path("dataset/benchmark_cifake/test/fake/cifake_fake_000510.png")},
    {"filename": "cifake_real_000510.png", "label": "real", "path": Path("dataset/benchmark_cifake/test/real/cifake_real_000510.png")},
]

print(f"{'Filename':<22} | {'Label':<4} | {'SRM Energy':<11} | {'Lap Energy':<11} | {'Lap Kurtosis':<12} | {'HF/LF (x1e-3)':<14} | {'HF Energy %':<12}")
print("-" * 95)

real_metrics = []
fake_metrics = []

for item in benchmark_items:
    img_path = benchmark_dir / item["label"] / item["filename"]
    if not img_path.exists():
        continue
    with Image.open(img_path) as im:
        feats = analyze_frequency_features(im)
    
    label = item["label"]
    if label == "real":
        real_metrics.append(feats)
    else:
        fake_metrics.append(feats)
        
    print(f"{item['filename']:<22} | {label:<4} | {feats['srm_energy']:<11.5f} | {feats['lap_energy']:<11.5f} | {feats['lap_kurtosis']:<12.2f} | {feats['hf_to_lf_ratio']:<14.3f} | {feats['hf_energy_pct']:<12.4f}%")

for extra in extra_samples:
    if not extra["path"].exists():
        continue
    with Image.open(extra["path"]) as im:
        feats = analyze_frequency_features(im)
    if extra["label"] == "real":
        real_metrics.append(feats)
    else:
        fake_metrics.append(feats)
    print(f"{extra['filename']:<22} | {extra['label']:<4} | {feats['srm_energy']:<11.5f} | {feats['lap_energy']:<11.5f} | {feats['lap_kurtosis']:<12.2f} | {feats['hf_to_lf_ratio']:<14.3f} | {feats['hf_energy_pct']:<12.4f}%")

print("-" * 95)

# Group statistics
print("\n" + "=" * 80)
print("STATISTICAL SUMMARY: GENUINE PHOTOGRAPHS VS SYNTHETIC / DEEPFAKE IMAGES")
print("=" * 80)

def print_stat(name, key):
    r_vals = [m[key] for m in real_metrics]
    f_vals = [m[key] for m in fake_metrics]
    print(f"Feature: {name:<25}")
    print(f"  Genuine Real (Mean ± Std):   {np.mean(r_vals):.5f} ± {np.std(r_vals):.5f}  [Min: {np.min(r_vals):.5f}, Max: {np.max(r_vals):.5f}]")
    print(f"  Synthetic Fake (Mean ± Std): {np.mean(f_vals):.5f} ± {np.std(f_vals):.5f}  [Min: {np.min(f_vals):.5f}, Max: {np.max(f_vals):.5f}]")
    # Simple threshold separation test (ROC-AUC like)
    overlap = max(0, min(np.max(r_vals), np.max(f_vals)) - max(np.min(r_vals), np.min(f_vals)))
    print(f"  Range Overlap:               {overlap:.5f}\n")

print_stat("SRM Residual Energy", "srm_energy")
print_stat("Laplacian High-Pass Energy", "lap_energy")
print_stat("Laplacian Kurtosis (Spikes)", "lap_kurtosis")
print_stat("HF to LF Energy Ratio", "hf_to_lf_ratio")
print_stat("High Frequency Energy %", "hf_energy_pct")
