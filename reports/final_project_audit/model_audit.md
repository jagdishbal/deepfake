# Model Audit Report — Deepfake Analysis System

**Audit Date:** November 10, 2026  
**Auditor:** Senior AI/ML Engineer & Neural Architectures Lead  
**Scope:** Verification of all active production model checkpoints, architectures, parameter counts, and calibration parameters in `models/`  

---

## 1. Production Model Summary Matrix

| Modality | Model Version | Checkpoint File | Format | File Size | Parameters | Input Dimensions | Decision Threshold |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Image** | **Model V3** (Primary) | `models/saved_models/image/best_model_v3.keras` | Keras v3 SavedModel | 31,415,411 bytes | 2,984,162 | `[(None, 224, 224, 3), (None, 224, 224, 3)]` | **0.3800** |
| **Image** | **Model V2** (Fallback) | `models/saved_models/image/best_model_v2.h5` | HDF5 Keras 2 | 22,903,824 bytes | 2,259,265 | `(None, 224, 224, 3)` | **0.4800** |
| **Video** | **Model V2** (Bi-GRU) | `models/saved_models/video/best_model.h5` | HDF5 Keras 2 | 10,429,096 bytes | 2,512,385 | `(None, 5, 224, 224, 3)` | **0.4000** |
| **Audio** | **Model V2** (140-dim MLP) | `models/saved_models/audio/best_model.h5` | HDF5 Keras 2 | 173,656 bytes | 11,418 | `(None, 140)` | **0.3000** |

---

## 2. In-Depth Architectural Inspections

### A. Image Model V3 (Dual-Branch Spatial-Frequency Architecture)
- **File:** `models/saved_models/image/best_model_v3.keras`
- **Architecture Strategy:**
  - Standard CNNs often overfit to color distribution and semantic content. Model V3 captures high-frequency residual artifacts introduced by generative upsampling (GAN checkerboard artifacts and diffusion PRNU disruption).
  - **Branch 1 (Global Spatial):** MobileNetV2 pretrained backbone processing full image resized to $224 \times 224$.
  - **Branch 2 (Focal High-Frequency):** Custom `SRMResidualLayer` applying spatial rich model high-pass filter kernels to a center 60% crop, followed by convolutional feature extraction.
  - **Fusion Head:** Concatenates global spatial embeddings with frequency residual embeddings, passing through a 128-dim dense layer with Dropout(0.3) and final sigmoid output.
- **Threshold Calibration:**
  - Configured in `models/calibration_v3.json`: Optimal decision threshold = **0.3800**.
  - Output is $P(\text{Real})$. If $P(\text{Real}) \ge 0.3800 \implies \text{Real}$, otherwise $\text{Fake}$.

### B. Image Model V2 (Fallback Architecture)
- **File:** `models/saved_models/image/best_model_v2.h5`
- **Architecture:** Standard MobileNetV2 feature extractor + GlobalAveragePooling2D + Dropout(0.2) + Dense(1, sigmoid).
- **Threshold Calibration:** Configured in `models/calibration.json`: Threshold = **0.4800** (uncertainty window `[0.42, 0.54]`).
- **Gating Logic:** Invoked automatically if an image has extreme aspect ratios ($> 3.0$), low resolution ($< 512 \times 320$), heavy JPEG lossy compression ($\text{BPP} < 0.04$), or if the V3 model is unavailable.

### C. Video Model V2 (TimeDistributed MobileNetV2 + Bi-GRU)
- **File:** `models/saved_models/video/best_model.h5`
- **Architecture Details:**
  - **Spatial Feature Extractor:** `TimeDistributed(MobileNetV2)` converting each frame into a 1280-dimensional feature embedding.
  - **Temporal Sequence Modeling:** Bidirectional GRU (`Bidirectional(GRU(64, return_sequences=False))`) modeling temporal continuity across frames.
  - **Classification Head:** Dropout(0.4) + Dense(32, ReLU) + Dense(1, sigmoid).
- **Key Inspection Finding:**
  - The model was trained with `NUM_FRAMES = 5`.
  - Passing 10 frames into this 5-frame recurrent unit previously caused temporal state drift, which inflated $P(\text{Real})$ and misclassified fakes as real.
  - Corrected in `predict.py` to extract temporal windows of exactly 5 frames each.
- **Threshold Calibration:**
  - Updated in `models/calibration.json`: Threshold = **0.4000**, uncertainty window = `[0.32, 0.48]`.

### D. Audio Model V2 (140-Dimensional Acoustic Feature MLP)
- **File:** `models/saved_models/audio/best_model.h5`
- **Architecture Details:**
  - Input Layer: `(None, 140)` matching the 140 extracted acoustic features:
    - 40 MFCC statistics (20 mean, 20 std)
    - 40 Delta-MFCC statistics (20 mean, 20 std)
    - 40 Delta-2 MFCC statistics (20 mean, 20 std)
    - 14 Spectral Contrast statistics (7 mean, 7 std)
    - 2 Spectral Rolloff statistics (1 mean, 1 std)
    - 2 Spectral Flatness statistics (1 mean, 1 std)
    - 2 Zero Crossing Rate statistics (1 mean, 1 std)
  - Layer 1: `acoustic_norm (Normalization)` layer (281 non-trainable feature normalization parameters).
  - Layer 2: `Dense(64, ReLU)` with L2 regularization ($10^{-4}$).
  - Layer 3: `Dropout(0.3)`.
  - Layer 4: `Dense(32, ReLU)` with L2 regularization ($10^{-4}$).
  - Layer 5: `Dropout(0.2)`.
  - Layer 6: `Dense(1, sigmoid)` (`probability_real_output`).
- **Threshold Calibration:**
  - Configured in `models/calibration.json`: Threshold = **0.3000**, uncertainty window = `[0.22, 0.38]`.

---

## 3. Thread Synchronization & Concurrency Safety

All production models are cached using `@lru_cache(maxsize=4)`. To prevent HDF5 race conditions and segmentation faults under concurrent requests, `predict.py` enforces thread synchronization:

```python
_MODEL_LOAD_LOCK = threading.Lock()

@lru_cache(maxsize=4)
def _load_model(modality: str):
    with _MODEL_LOAD_LOCK:
        # Load and return checkpoint safely
```

This guarantees that multiple HTTP request threads can execute inference concurrently without file-descriptor collision or memory access violations.
