# Stage 6 Resolution-Gated Dual Pipeline Bug Fix Report

## 1. Executive Summary
Following the integration of Stage 6 (Resolution-Gated Dual Pipeline) into the Deepfake Detection web platform, image detection failed or caused unexpected page reloads during image upload and prediction. An exhaustive root-cause investigation across route handlers, model loaders, preprocessing layers, database schemas, and Jinja2 rendering templates was conducted. All underlying bugs were resolved with 100% backward compatibility preserved. 

All 24 pytest unit tests and end-to-end integration workflows (high-res V3, low-res V2, compressed fallback, and edge cases) were verified with complete success.

---

## 2. Root Cause Analysis

1. **Missing `_preprocess_image` Alias in `predict.py`**:
   - In Stage 6, `_preprocess_image` was refactored into `_preprocess_image_v2` without maintaining a backward-compatible alias.
   - Any external callers, unit tests (`test_image.py`), and evaluation scripts invoking `predict._preprocess_image` crashed with `AttributeError: module 'predict' has no attribute '_preprocess_image'`.

2. **File Path Type Coercion in Routing and Preprocessing**:
   - `route_image(file_path)` and `predict_image(file_path)` called `.stat().st_size` assuming `file_path` was always a `Path` object. When passed strings from web upload pipelines or third-party callers, an `AttributeError: 'str' object has no attribute 'stat'` was thrown and caught by a generic fallback, erroneously misrouting high-resolution images to Model V2.

3. **Windows Open File Handles & Locked File Exceptions in PIL**:
   - PIL `Image.open(file_path).convert("RGB")` was invoked without a context manager (`with Image.open(file_path) as img:`). On Windows, this retained open file descriptors on disk.
   - If an exception occurred during inference, `_process_upload()` called `file_path.unlink(missing_ok=True)`. On Windows, unlinking open files raises `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process`, crashing Flask and masking the actual validation error.

4. **Non-Scalar Prediction Output Validation**:
   - `predict_image()` checked `raw_output[0]` without verifying `raw_output.size == 1`. When models returned non-scalar outputs (such as `[[0.2, 0.8]]`), the invalid shape check failed to trigger, violating the strict API contract tested in `test_image.py`.

5. **Checkpoint Mock Leakage in `_load_model`**:
   - `_load_model` resolved `calib_checkpoint` using hardcoded paths relative to `__file__`, ignoring `MODEL_DIR` when patched by unit test mocks (`test_missing_checkpoint_fails_explicitly`). Candidate checkpoints are now resolved relative to `MODEL_DIR`.

6. **History Template Column Mapping**:
   - In `history.html`, table cells referred to `{{ item.file_name }}` and `{{ item.date }}`, whereas the database columns are `filename` and `uploaded_at`. Aliasing these in `app.py` restored display consistency without requiring any frontend redesign.

---

## 3. Files Modified & Fixes Applied

### `predict.py`
- Added backward-compatible alias `_preprocess_image = _preprocess_image_v2`.
- Ensured all functions (`route_image`, `predict_image`, `_preprocess_image_v2`, `_preprocess_image_v3`) coerce input to `Path(file_path)`.
- Wrapped image decodes in `with Image.open(file_path) as img:` to ensure immediate file descriptor closure.
- Added strict output dimension verification:
  ```python
  if (
      raw_output.size != 1
      or not np.all(np.isfinite(raw_output))
      or np.any((raw_output < 0.0) | (raw_output > 1.0))
  ):
      raise PredictionError("The model returned an invalid prediction score.")
  ```
- Scoped calibration checkpoint candidates inside `_load_model` to `MODEL_DIR`.

### `app.py`
- Protected upload unlinks in `_process_upload()` inside `try...except` so file lock errors never prevent user flash alerts from displaying.
- Coerced `confidence=float(result["confidence"])` before passing to `result.html` rendering.
- Aliased `filename AS file_name` and `uploaded_at AS date` in `/history` query to restore filename and timestamp visibility in the history table.

---

## 4. Test Results

### Pytest Unit Test Suite
- **Executed**: `pytest tests/`
- **Result**: `24 passed in 10.35s` (100% pass rate).

### End-to-End Image Workflow Verification
- **High-Res Route (V3)**: `ai-generated-8622236_1280.jpg` (1280x1280, BPP 0.331) -> Routed to Model V3, threshold 0.3800, classified as Fake (100.00% confidence). Result page and audit record confirmed.
- **Low-Res Route (V2)**: `OIP.jpg` (220x220, max < 512) -> Routed to Model V2, threshold 0.5200, classified as Fake (86.35% confidence). Result page and audit record confirmed.
- **Compressed Fallback**: `heavily_compressed.jpg` (BPP 0.0178 < 0.04) -> Routed to Model V2, threshold 0.5200.
- **Corrupt Image**: `b"corrupted image content"` -> Rejected with danger flash message ("The uploaded image is invalid or could not be decoded.").
- **Empty File**: 0 bytes -> Rejected with danger flash message ("The uploaded file is empty.").
- **Unsupported Format**: `.txt` -> Rejected with danger flash message ("That file format is not supported for image uploads.").
- **History View**: Complete history listing with correct timestamps and links to prediction results.
