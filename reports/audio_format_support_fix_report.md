# Audio Format Support & Compatibility Fix Report

## 1. Executive Summary
The Audio Detection feature in the Deepfake Analysis web platform was investigated and enhanced to support full upload compatibility for **WAV**, **MP3**, and **MP4** audio files. The existing audio detection model (`models/saved_models/audio/best_model.h5`), model weights, prediction logic, calibration thresholds (`0.4900`), and database schema remain 100% unchanged.

All 9 required test scenarios (including valid audio uploads, missing audio streams in MP4 files, corrupt files, unsupported formats, empty files, and duration/size overages) were verified end-to-end through the complete upload → format conversion → preprocessing → prediction → result → history lifecycle. All 24 existing pytest unit tests pass with zero regressions.

---

## 2. Root Cause Analysis

### Why MP4 Files Were Rejected
1. **Frontend Input Filter**: The HTML `<input type="file" name="audio">` in `templates/upload_audio.html` only specified `accept=".wav,.mp3,.aac,.m4a"`. Browsers explicitly filtered out `.mp4` files from the file picker.
2. **Backend Route Rejection**: In `predict.py`, `SUPPORTED_EXTENSIONS["audio"]` did not include `".mp4"`. When `_save_upload()` evaluated the file extension in `app.py`, it threw an `UploadValidationError("That file format is not supported for audio uploads.")`.
3. **Container Format Incompatibility in Preprocessing**: The underlying audio preprocessing function `extract_audio_features()` in `preprocessing/clean_audio.py` relied solely on `librosa.load()` / `soundfile.read()`. `soundfile` and `libsndfile` cannot demux MP4 container files (`LibsndfileError: Format not recognised`), failing immediately without a dedicated media extractor.
4. **Missing Audio Stream Handling**: Video files (such as silent MP4 videos with no audio streams) were not probed, leading to decoding crashes rather than user-friendly validation feedback.

### Why MP3 Files Faced Inconsistencies
1. **Decoder Backend Dependency**: `librosa.load()` defaults to `soundfile`, which in turn relies on `libsndfile`. On Windows without system FFmpeg on the system `PATH`, `libsndfile` frequently fails on MP3s with ID3v2 tags, variable bitrates (VBR), or proprietary headers. When `soundfile` failed, `librosa` attempted fallback to `audioread`, which crashed because `shutil.which('ffmpeg')` returned `None`.
2. **Lack of Standardized Transcoding**: Without a dedicated conversion step into the model's expected 16 kHz mono 16-bit PCM format, non-standard channel layouts (stereo/surround) or sample rates (44.1 kHz, 48 kHz) risked decoding errors or subtle feature skew.

---

## 3. Files Modified

| File | Changes Applied |
|---|---|
| `templates/upload_audio.html` | Updated `accept` attribute to include `.wav,.mp3,.mp4,.aac,.m4a,audio/*,video/mp4`. Updated helper text and instructions to document WAV, MP3, and MP4 support. |
| `preprocessing/clean_audio.py` | Added `probe_audio()` to inspect stream types and duration via FFmpeg. Added `extract_audio_track_ffmpeg()` to transcode/demux any media format into standardized 16 kHz mono 16-bit PCM WAV. Added missing audio track detection for MP4, duration validation (maximum 300 seconds), and safe temporary file lifecycle management. |
| `predict.py` | Added `".mp4"` to `SUPPORTED_EXTENSIONS["audio"]`. Enhanced `_preprocess_audio()` to catch and surface specific user-friendly decode and missing audio track errors. |
| `app.py` | Added `ALLOWED_MIMETYPES` mapping and MIME-type validation in `_save_upload()` to reject disguised/non-media files while safely permitting `audio/*` and `video/mp4` audio uploads. |

---

## 4. Libraries Used & FFmpeg Availability

- **FFmpeg Engine**: Utilized `imageio_ffmpeg.get_ffmpeg_exe()` which provides a self-contained, pre-compiled static FFmpeg binary:
  - Binary: `ffmpeg version 7.1-essentials_build-www.gyan.dev`
  - Path: `C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe`
  - Subprocess safety: Always invoked with `-nostdin` and `stdin=subprocess.DEVNULL` to prevent process hangs on Windows.
- **Audio Processing**:
  - `soundfile` (v0.14.0): For reading standardized PCM WAV audio data into 1D `float32` numpy arrays.
  - `librosa` (v0.10.2): For computing Mel-spectrograms (`n_mels=128`, `fmax=8000`) and power-to-dB conversion.
  - `numpy`: For waveform zero-padding, length truncation (to 48,000 samples for 3.0s at 16 kHz), and min-max normalization.

---

## 5. Conversion Method & Architecture

```
Uploaded Audio (WAV / MP3 / MP4)
              │
              ▼
   MIME & Extension Validation (_save_upload in app.py)
              │
              ▼
   Stream Probe via FFmpeg (probe_audio in clean_audio.py)
   ├── Check: Audio stream present? (If no -> ValueError: Missing audio track)
   └── Check: Duration <= 300s? (If >300s -> ValueError: Duration exceeded)
              │
              ▼
   Audio Transcoding & Demuxing (extract_audio_track_ffmpeg)
   ├── Input: Any valid WAV, MP3, or MP4
   ├── Filters: -vn (drop video), -ac 1 (mono downmix), -ar 16000 (16 kHz)
   ├── Codec: -acodec pcm_s16le (16-bit signed PCM)
   └── Output: Temporary WAV in system temp directory
              │
              ▼
   Waveform Loading & Mel-Spectrogram Extraction
   ├── Read via soundfile: 1D float32 array @ 16,000 Hz
   ├── Unlink temporary WAV in finally block (guaranteed cleanup)
   ├── Slice / Zero-pad to exactly 3.0 seconds (48,000 samples)
   └── Compute Mel-spectrogram: shape (128, 94, 1), normalized to [0, 1]
              │
              ▼
   Model Inference (models/saved_models/audio/best_model.h5)
   └── Sigmoid prediction with calibrated threshold = 0.4900
```

---

## 6. Actual Test Results

All tests were executed against the active MySQL database and Flask test client in `test_audio_format_support.py`.

### Direct Pipeline Verification
| Test Case | Format / Condition | Expected Behavior | Actual Result | Status |
|---|---|---|---|---|
| **Case 1** | WAV Audio (`valid_sample.wav`) | Standard inference | Pred: Fake, Conf: 51.01%, Modality: audio | **PASSED** |
| **Case 2** | MP3 Audio (`valid_sample.mp3`) | Decoded & converted to 16kHz mono | Pred: Fake, Conf: 51.01%, Modality: audio | **PASSED** |
| **Case 3** | MP4 with Audio (`valid_with_audio.mp4`) | Audio track extracted & processed | Pred: Fake, Conf: 51.01%, Modality: audio | **PASSED** |
| **Case 4** | MP4 Silent (`silent_no_audio.mp4`) | Missing track detected | Rejected with `The uploaded MP4 file does not contain an audio track.` | **PASSED** |
| **Case 5** | Corrupted MP3 (`corrupted_audio.mp3`) | Decode error caught | Rejected with `The uploaded audio could not be decoded. Check its format and codec.` | **PASSED** |
| **Case 6** | Corrupted MP4 (`corrupted_video.mp4`) | Container decode error caught | Rejected with `The uploaded audio could not be decoded.` | **PASSED** |
| **Case 7** | Audio > 300s (`over_duration.mp3`) | Duration limit enforced | Rejected with `The audio duration (310.1s) exceeds the allowed limit of 300 seconds.` | **PASSED** |

### End-to-End Web Lifecycle Verification (Upload → Prediction → Result → History)
| Test Scenario | Upload Data | HTTP Status | Response Details | Status |
|---|---|---|---|---|
| **Web 1: WAV Upload** | `test_audio.wav` (`audio/wav`) | 302 → `/result/98` (200) | Classified as Deepfake, confidence stored in MySQL, audit log created | **PASSED** |
| **Web 2: MP3 Upload** | `test_audio.mp3` (`audio/mpeg`) | 302 → `/result/99` (200) | Audio converted, classified as Deepfake, audit log created | **PASSED** |
| **Web 3: MP4 Audio Upload** | `test_audio.mp4` (`video/mp4`) | 302 → `/result/100` (200) | Audio track extracted, classified as Deepfake, audit log created | **PASSED** |
| **Web 4: Silent MP4** | `test_silent.mp4` | 200 (redirect to form) | Danger alert: *"The uploaded MP4 file does not contain an audio track. Please upload media with audio."* | **PASSED** |
| **Web 5: Corrupt MP3** | `corrupt.mp3` | 200 (redirect to form) | Danger alert: *"The uploaded audio could not be decoded. Check its format and codec."* | **PASSED** |
| **Web 6: Corrupt MP4** | `corrupt.mp4` | 200 (redirect to form) | Danger alert: *"The uploaded audio could not be decoded."* | **PASSED** |
| **Web 7: Unsupported File** | `notes.txt` (`text/plain`) | 200 (redirect to form) | Danger alert: *"That file format is not supported for audio uploads."* | **PASSED** |
| **Web 8: Empty File** | `empty.mp3` (0 bytes) | 200 (redirect to form) | Danger alert: *"The uploaded file is empty."* | **PASSED** |
| **Web 9A: Duration Over Limit** | `too_long.mp3` (310s) | 200 (redirect to form) | Danger alert: *"The audio duration (310.1s) exceeds the allowed limit of 300 seconds (5 minutes)."* | **PASSED** |
| **Web 9B: Size Over Limit** | `too_large.wav` (51 MB) | 200 (redirect to form) | Danger alert: *"The audio file exceeds the 50 MB limit."* | **PASSED** |
| **Web 10: History Page** | GET `/history` | 200 | All three uploads (`test_audio.wav`, `test_audio.mp3`, `test_audio.mp4`) properly rendered with action links | **PASSED** |

### Unit Test Suite
- Command: `pytest tests/`
- Result: **24 passed in 11.48s** (100% pass rate).

---

## 7. Confirmation of Preservation Rules

1. **Audio Model Unchanged**: `models/saved_models/audio/best_model.h5` was neither retrained nor overwritten. Model weights and layer configurations are untouched.
2. **Calibration Threshold Unchanged**: The calibrated decision threshold for audio remains exactly `0.4900` as specified in `models/calibration.json`.
3. **Prediction Logic Unchanged**: Binary classification ("Real" vs "Fake"), model score calculations, and label mappings are identical to baseline.
4. **Image & Video Modalities Preserved**: Image detection (including Stage 6 Resolution-Gated Dual Pipeline V2/V3) and video detection pipelines were not modified.
5. **Database Schema Preserved**: No MySQL tables or columns were altered (`predictions` table schema remains intact).
6. **Frontend Preserved**: No layouts, navigation bars, or styling were redesigned; only the file input `accept` attribute was updated.
7. **Security Protections Maintained**: Session authentication, file size enforcement, `secure_filename` sanitization, unique UUID prefixing, and path traversal defenses remain active.
