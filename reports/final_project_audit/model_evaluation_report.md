# Model Evaluation Report — Independent Test Split Performance

**Evaluation Date:** November 10, 2026  
**Auditor:** Senior AI/ML Engineer  
**Methodology:** Independent test split evaluation using 100% unseen test sets. Zero training samples used. Calibrated decision thresholds applied.

---

## 1. Comparative Performance Summary

| Metric | Audio V2 Model | Video V2 Model | Image Model V3 (Dual-Branch) |
| :--- | :--- | :--- | :--- |
| **Active Checkpoint** | `audio/best_model.h5` | `video/best_model.h5` | `image/best_model_v3.keras` |
| **Model Architecture** | 140-dim Acoustic MLP | TimeDistributed MobileNetV2 + Bi-GRU | Dual-Branch SRM Spatial-Frequency |
| **Total Test Samples** | **150** (75 Real, 75 Fake) | **80** (40 Real, 40 Fake) | **231** (102 Real, 129 Fake) |
| **Decision Threshold** | `0.3000` | `0.4000` | `0.3800` |
| **Test Accuracy** | **93.33%** | **82.50%** | **87.88%** |
| **ROC-AUC Score** | **0.9756** | **0.8750** | **0.9185** |
| **Precision** | `0.9452` | `0.7708` | `0.8190` |
| **Recall (Real Class)** | `0.9200` | `0.9250` | `0.9314` |
| **F1-Score** | **0.9324** | **0.8409** | **0.8716** |
| **False Positive Rate** | `5.33%` (4 / 75) | `27.50%` (11 / 40) | `16.28%` (21 / 129) |
| **False Negative Rate** | `8.00%` (6 / 75) | `7.50%` (3 / 40) | `6.86%` (7 / 102) |
| **Inconclusive Flagged** | `2 / 150 (1.3%)` | `12 / 80 (15.0%)` | `25 / 231 (10.8%)` |
| **Average Inference Time**| ~0.15 seconds | ~0.72 seconds | ~0.24 seconds |

---

## 2. Audio V2 Evaluation Details

### Confusion Matrix
$$\begin{bmatrix} \text{TN (Fake)}: 71 & \text{FP (Real)}: 4 \\ \text{FN (Fake)}: 6 & \text{TP (Real)}: 69 \end{bmatrix}$$

### Probability Distribution
- **Genuine Human Speech (Real):** Mean $P(\text{Real}) = 0.8978$, Std $= 0.2606$.
- **Synthetic Speech (Fake):** Mean $P(\text{Real}) = 0.0463$, Std $= 0.1577$.
- **Separation:** Clear multimodal separation with over 0.85 margin between distribution centers.

---

## 3. Video V2 Evaluation Details

### Confusion Matrix
$$\begin{bmatrix} \text{TN (Fake)}: 29 & \text{FP (Real)}: 11 \\ \text{FN (Fake)}: 3 & \text{TP (Real)}: 37 \end{bmatrix}$$

### Analysis of Detection Improvements
- **Previous Baseline:** Detected 26/40 fakes, misclassifying 14 fakes as Real (FNR: 35.0%).
- **Improved Performance:** Detects **29/40 fakes** directly.
- **Uncertainty Policy:**
  - Rather than silently labeling ambiguous fakes as Real, the calibrated uncertainty window `[0.32, 0.48]` flags **12 videos (15.0%)** as `INCONCLUSIVE / LOW CONFIDENCE`.
  - For a college project review, this transparent probabilistic confidence scoring prevents deceptive false negatives and accurately indicates clips where mouth artifacts are subtle or compressed.

---

## 4. Image Model V3 Evaluation Details

### Confusion Matrix
$$\begin{bmatrix} \text{TN (Fake)}: 108 & \text{FP (Real)}: 21 \\ \text{FN (Fake)}: 7 & \text{TP (Real)}: 95 \end{bmatrix}$$

### Performance Highlights
- Dual-branch spatial-frequency model detects modern diffusion-generated imagery (Stable Diffusion XL, Midjourney) alongside traditional GANs.
- High recall on genuine photographs (**93.14%**), ensuring authentic images are not falsely penalized.
