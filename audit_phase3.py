import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import numpy as np
import scipy.stats as stats
from sklearn.metrics import roc_auc_score

with open("reports/phase3_experimental_validation_results.json", "r") as f:
    results_data = json.load(f)

samples = results_data["individual_data"]
print("=" * 80)
print(f"STEP 1 AUDIT: VERIFYING EXPERIMENTAL DATA (N={len(samples)})")
print("=" * 80)

# Check categories and counts
cat_counts = {}
for s in samples:
    c = s["category"]
    cat_counts[c] = cat_counts.get(c, 0) + 1
print("Sample distribution by category:")
for c, cnt in cat_counts.items():
    real_c = sum(1 for s in samples if s["category"] == c and s["y_true"] == 1)
    fake_c = sum(1 for s in samples if s["category"] == c and s["y_true"] == 0)
    print(f"  {c:<25}: {cnt:>2} samples (Real: {real_c:>2}, Fake: {fake_c:>2})")

total_real = sum(1 for s in samples if s["y_true"] == 1)
total_fake = sum(1 for s in samples if s["y_true"] == 0)
print(f"\nOverall Class Counts: Real = {total_real}, Fake = {total_fake}, Total = {len(samples)}")

print("\nDetailed breakdown of 14 High-Res Samples:")
print(f"{'Filename':<20} | {'True Label':<10} | {'SRM Energy':<12} | {'App A P(Real)':<14} | {'App B P(Real)':<14} | {'App C P(Real)':<14} | {'App D P(Real)'}")
print("-" * 105)
for s in samples[:14]:
    lbl = "REAL (1)" if s["y_true"] == 1 else "FAKE (0)"
    print(f"{s['name']:<20} | {lbl:<10} | {s['srm_energy']:<12.5f} | {s['p_approach_a']:<14.4f} | {s['p_approach_b']:<14.4f} | {s['p_approach_c']:<14.4f} | {s['p_approach_d']:.4f}")

print("\n" + "=" * 80)
print("STEP 2 AUDIT: RECOMPUTING METRICS UNDER USER STANDARD DEFINITIONS")
print("=" * 80)
print("Definitions:")
print("  Positive Class = FAKE (0 in code)")
print("  Negative Class = REAL (1 in code)")
print("  TP: Actual FAKE, Predicted FAKE")
print("  TN: Actual REAL, Predicted REAL")
print("  FP: Actual REAL, Predicted FAKE  (False Alarm / Type I error)")
print("  FN: Actual FAKE, Predicted REAL  (Missed Deepfake / Type II error)")
print("  Threshold: 0.5200 (If P(Real) >= 0.52 -> Pred REAL; Else Pred FAKE)")
print("-" * 80)

def compute_user_metrics(sample_subset, prob_key, threshold=0.52):
    # In code, p is P(Real).
    # If p < threshold -> Predicted FAKE (Positive)
    # If p >= threshold -> Predicted REAL (Negative)
    # y_true is 1 for REAL, 0 for FAKE
    
    tp = 0 # actual fake (0), pred fake (p < thresh)
    tn = 0 # actual real (1), pred real (p >= thresh)
    fp = 0 # actual real (1), pred fake (p < thresh)
    fn = 0 # actual fake (0), pred real (p >= thresh)
    
    y_true_fake_pos = [] # 1 if FAKE, 0 if REAL
    y_score_fake_pos = [] # probability of FAKE = 1 - p
    
    for s in sample_subset:
        p_real = s[prob_key]
        is_actual_fake = (s["y_true"] == 0)
        is_pred_fake = (p_real < threshold)
        
        y_true_fake_pos.append(1 if is_actual_fake else 0)
        y_score_fake_pos.append(1.0 - p_real)
        
        if is_actual_fake and is_pred_fake:
            tp += 1
        elif (not is_actual_fake) and (not is_pred_fake):
            tn += 1
        elif (not is_actual_fake) and is_pred_fake:
            fp += 1
        elif is_actual_fake and (not is_pred_fake):
            fn += 1
            
    total = len(sample_subset)
    actual_fake = tp + fn
    actual_real = tn + fp
    
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0 # Precision of detecting Fakes
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0    # Recall / Detection rate of Fakes
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    fpr = fp / actual_real if actual_real > 0 else 0.0   # Real misclassified as Fake (FP / (FP + TN))
    fnr = fn / actual_fake if actual_fake > 0 else 0.0   # Fake misclassified as Real (FN / (FN + TP))
    
    try:
        auc = roc_auc_score(y_true_fake_pos, y_score_fake_pos)
    except:
        auc = 0.5
        
    return {
        "total": total,
        "actual_fake": actual_fake,
        "actual_real": actual_real,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": fpr,
        "fnr": fnr,
        "auc": auc
    }

approaches = [
    ("Approach A: Existing best_model_v2.h5 (Global)", "p_approach_a"),
    ("Approach B: Multi-Scale Preprocessing (Weighted)", "p_approach_b"),
    ("Approach B-Min: Multi-Scale (Max-Fake Crop)", "p_approach_b_min"),
    ("Approach C: Frequency-Domain (SRM Residuals)", "p_approach_c"),
    ("Approach D: Combined Multi-Scale + Frequency", "p_approach_d"),
]

# 1. High-Res Subset (N=14: 10 Real, 4 Fake)
hr_samples = [s for s in samples if s["category"] == "high_res"]
print(f"\n1. HIGH-RESOLUTION BENCHMARK (N=14: Actual Fake = 4, Actual Real = 10):")
print(f"{'Approach':<45} | TP | TN | FP | FN | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'ROC-AUC':<7} | {'FPR (FP/Real)':<13} | {'FNR (FN/Fake)'}")
print("-" * 130)
for name, key in approaches:
    m = compute_user_metrics(hr_samples, key)
    print(f"{name:<45} | {m['tp']:>2} | {m['tn']:>2} | {m['fp']:>2} | {m['fn']:>2} | {m['accuracy']*100:5.1f}% | {m['precision']*100:5.1f}% | {m['recall']*100:5.1f}% | {m['f1']*100:5.1f}% | {m['auc']:7.4f} | {m['fpr']*100:11.1f}% | {m['fnr']*100:11.1f}%")

# 2. Composite Benchmark (N=54: 24 Fake, 30 Real)
print(f"\n2. COMPOSITE BENCHMARK (N=54: Actual Fake = 24, Actual Real = 30):")
print(f"{'Approach':<45} | TP | TN | FP | FN | {'Acc':<6} | {'Prec':<6} | {'Rec':<6} | {'F1':<6} | {'ROC-AUC':<7} | {'FPR (FP/Real)':<13} | {'FNR (FN/Fake)'}")
print("-" * 130)
for name, key in approaches:
    m = compute_user_metrics(samples, key)
    print(f"{name:<45} | {m['tp']:>2} | {m['tn']:>2} | {m['fp']:>2} | {m['fn']:>2} | {m['accuracy']*100:5.1f}% | {m['precision']*100:5.1f}% | {m['recall']*100:5.1f}% | {m['f1']*100:5.1f}% | {m['auc']:7.4f} | {m['fpr']*100:11.1f}% | {m['fnr']*100:11.1f}%")

print("\n" + "=" * 80)
print("STEP 3 AUDIT: STATISTICAL RIGOR OF FREQUENCY-DOMAIN FEATURES")
print("=" * 80)

# Statistical tests on SRM and DCT
srm_reals = [s["srm_energy"] for s in hr_samples if s["y_true"] == 1]
srm_fakes = [s["srm_energy"] for s in hr_samples if s["y_true"] == 0]

t_stat, p_val = stats.ttest_ind(srm_fakes, srm_reals, equal_var=False)
u_stat, u_pval = stats.mannwhitneyu(srm_fakes, srm_reals, alternative="greater")

# Cohen's d
diff = np.mean(srm_fakes) - np.mean(srm_reals)
pooled_std = np.sqrt(((len(srm_fakes)-1)*np.var(srm_fakes, ddof=1) + (len(srm_reals)-1)*np.var(srm_reals, ddof=1)) / (len(srm_fakes) + len(srm_reals) - 2))
cohens_d = diff / pooled_std

print(f"High-Res SRM Residual Energy Analysis (N_real={len(srm_reals)}, N_fake={len(srm_fakes)}):")
print(f"  Real Mean +/- Std:  {np.mean(srm_reals):.5f} +/- {np.std(srm_reals, ddof=1):.5f}")
print(f"  Fake Mean +/- Std:  {np.mean(srm_fakes):.5f} +/- {np.std(srm_fakes, ddof=1):.5f}")
print(f"  Welch's t-statistic: {t_stat:.4f} (p-value: {p_val:.5f})")
print(f"  Mann-Whitney U:     {u_stat:.1f} (p-value: {u_pval:.5f})")
print(f"  Cohen's d:          {cohens_d:.4f} (Effect size)")

# Also on CIFAKE (32x32)
srm_cifake_reals = [s["srm_energy"] for s in samples if s["category"] == "standard_cifar_real"]
srm_cifake_fakes = [s["srm_energy"] for s in samples if s["category"] == "standard_ai_cifake"]

print(f"\nCIFAKE (32x32) SRM Residual Energy Analysis (N_real={len(srm_cifake_reals)}, N_fake={len(srm_cifake_fakes)}):")
print(f"  Real Mean +/- Std:  {np.mean(srm_cifake_reals):.5f} +/- {np.std(srm_cifake_reals, ddof=1):.5f}")
print(f"  Fake Mean +/- Std:  {np.mean(srm_cifake_fakes):.5f} +/- {np.std(srm_cifake_fakes, ddof=1):.5f}")
t_stat_ci, p_val_ci = stats.ttest_ind(srm_cifake_fakes, srm_cifake_reals, equal_var=False)
print(f"  Welch's t-statistic: {t_stat_ci:.4f} (p-value: {p_val_ci:.5f})")
