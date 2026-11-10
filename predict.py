import argparse
from functools import lru_cache
import json
import logging
from pathlib import Path
import threading

from config import MODEL_DIR

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from models.audio_model import build_audio_model
from models.image_model import build_image_model
from models.image_model_v3 import SRMResidualLayer, load_v3_image_model
from models.video_model import build_video_model
from preprocessing.clean_audio import extract_audio_features
from preprocessing.clean_video import extract_frames_from_video, extract_video_windows

MODEL_DIR = Path(MODEL_DIR)
logger = logging.getLogger(__name__)

class PredictionError(Exception):
    """Base exception for expected prediction failures."""


class UnsupportedMediaError(PredictionError):
    pass


class MediaDecodeError(PredictionError):
    pass


class ModelUnavailableError(PredictionError):
    pass


SUPPORTED_EXTENSIONS = {
    "image": {".jpg", ".jpeg", ".png", ".bmp", ".webp"},
    "video": {".mp4", ".avi", ".mov", ".mkv", ".webm"},
    "audio": {".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a", ".mp4"},
}

MODEL_BUILDERS = {
    "image": lambda: build_image_model(weights=None),
    "video": lambda: build_video_model(weights=None),
    "audio": build_audio_model,
}


def get_threshold(modality: str, model_version: str = "v2") -> float:
    """
    Retrieves the calibrated decision threshold.
    - For image with model_version 'v3': reads from models/calibration_v3.json (default 0.3800).
    - For all other modalities and V2: reads from models/calibration.json (default 0.5200 for image).
    """
    if modality.lower() == "image" and model_version.lower() == "v3":
        v3_calib_path = Path(__file__).parent / "models" / "calibration_v3.json"
        try:
            with open(v3_calib_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            val = data.get("optimal_threshold", data.get("active_threshold", 0.3800))
            if isinstance(val, (int, float)) and 0.0 <= val <= 1.0:
                return float(val)
        except Exception as exc:
            logger.warning("Could not read calibration_v3.json (%s), using 0.3800 fallback", exc)
        return 0.3800

    calib_path = Path(__file__).parent / "models" / "calibration.json"
    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise PredictionError("The prediction threshold configuration is unavailable or invalid.") from exc

    modality_settings = data.get(modality.lower()) if isinstance(data, dict) else None
    threshold = modality_settings.get("threshold") if isinstance(modality_settings, dict) else None
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not np.isfinite(threshold)
        or not 0.0 <= threshold <= 1.0
    ):
        raise PredictionError(f"The prediction threshold for {modality} is missing or invalid.")
    return float(threshold)


def _classify_probability(probability_real: float, modality: str, model_version: str = "v2") -> str:
    thresh = get_threshold(modality, model_version=model_version)
    return "Real" if probability_real >= thresh else "Fake"


def get_calibration_details(modality: str, model_version: str = "v2") -> dict:
    """Retrieves full calibration details (threshold, uncertainty range) for a modality."""
    thresh = get_threshold(modality, model_version=model_version)
    calib_path = Path(__file__).parent / "models" / "calibration.json"
    u_lower = max(0.0, round(thresh - 0.08, 2))
    u_upper = min(1.0, round(thresh + 0.08, 2))
    try:
        with open(calib_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mod_settings = data.get(modality.lower(), {})
        if "uncertainty_lower" in mod_settings:
            u_lower = float(mod_settings["uncertainty_lower"])
        if "uncertainty_upper" in mod_settings:
            u_upper = float(mod_settings["uncertainty_upper"])
    except Exception:
        pass
    return {
        "threshold": thresh,
        "uncertainty_lower": u_lower,
        "uncertainty_upper": u_upper,
    }


def compute_prediction_result(
    probability_real: float,
    threshold: float,
    uncertainty_lower: float,
    uncertainty_upper: float,
    modality: str,
    extra_fields: dict | None = None,
) -> dict:
    """Calculates true probabilities, proper confidence, and inconclusive state."""
    prediction = "Real" if probability_real >= threshold else "Fake"
    probability_fake = float(np.clip(1.0 - probability_real, 0.0, 1.0))
    probability_real = float(np.clip(probability_real, 0.0, 1.0))
    confidence = probability_real if prediction == "Real" else probability_fake

    # Flag prediction as inconclusive if probability is in the uncertainty window near the threshold or confidence < 0.60
    is_inconclusive = (uncertainty_lower <= probability_real <= uncertainty_upper) or (confidence < 0.60)
    confidence_status = "INCONCLUSIVE / LOW CONFIDENCE" if is_inconclusive else "CONFIDENT"
    dual_probabilities_text = f"REAL: {probability_real * 100:.1f}% | FAKE: {probability_fake * 100:.1f}%"

    res = {
        "prediction": prediction,
        "confidence": confidence,
        "probability_real": probability_real,
        "probability_fake": probability_fake,
        "is_inconclusive": is_inconclusive,
        "confidence_status": confidence_status,
        "dual_probabilities_text": dual_probabilities_text,
        "threshold": threshold,
        "modality": modality,
    }
    if extra_fields:
        res.update(extra_fields)
    return res


def detect_modality(file_path: str | Path, modality: str | None = None) -> str:
    extension = Path(file_path).suffix.lower()
    if modality is None:
        modality = next(
            (kind for kind, extensions in SUPPORTED_EXTENSIONS.items() if extension in extensions),
            None,
        )
    if modality not in SUPPORTED_EXTENSIONS:
        raise UnsupportedMediaError("This file type is not supported.")
    if extension not in SUPPORTED_EXTENSIONS[modality]:
        raise UnsupportedMediaError(f"That file format is not supported for {modality} uploads.")
    return modality


_MODEL_LOAD_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def _load_model(modality: str):
    """Loads the production V2 model (or video/audio models) with thread-safe locking."""
    with _MODEL_LOAD_LOCK:
        calib_checkpoint = None
        try:
            with open(Path(__file__).parent / "models" / "calibration.json", "r", encoding="utf-8") as f:
                calib_data = json.load(f)
                mod_data = calib_data.get(modality.lower(), {})
                if isinstance(mod_data, dict) and "checkpoint" in mod_data:
                    chk = Path(mod_data["checkpoint"])
                    if "saved_models" in chk.parts:
                        idx = chk.parts.index("saved_models")
                        chk_path = MODEL_DIR / Path(*chk.parts[idx + 1:])
                    else:
                        chk_path = MODEL_DIR / chk.name
                    if chk_path.is_file() and chk_path.stat().st_size > 0:
                        calib_checkpoint = chk_path
        except Exception:
            pass

        artifact_candidates = []
        if calib_checkpoint:
            artifact_candidates.append(calib_checkpoint)

        artifact_candidates.extend([
            MODEL_DIR / modality / "best_model_v2.h5",
            MODEL_DIR / modality / "best_model.h5",
            MODEL_DIR / f"{modality}_model.keras",
        ])
        artifact = next(
            (path for path in artifact_candidates if path.is_file() and path.stat().st_size > 0),
            None,
        )
        if artifact is None:
            raise ModelUnavailableError(
                f"The trained {modality} model is not available yet. Please try another media type."
            )

        try:
            return tf.keras.models.load_model(str(artifact), compile=False)
        except Exception:
            pass

        try:
            model = MODEL_BUILDERS[modality]()
            model.load_weights(str(artifact))
            return model
        except Exception as exc:
            raise ModelUnavailableError(
                f"The trained {modality} model could not be loaded. Please contact the administrator."
            ) from exc


@lru_cache(maxsize=1)
def _load_v3_image_model():
    """Loads the V3 Dual-Branch Spatial-Frequency model with thread-safe locking."""
    with _MODEL_LOAD_LOCK:
        v3_path = MODEL_DIR / "image" / "best_model_v3.keras"
        if not v3_path.is_file() or v3_path.stat().st_size == 0:
            raise ModelUnavailableError("The V3 image model checkpoint is not available.")
        try:
            return load_v3_image_model(v3_path)
        except Exception as exc:
            raise ModelUnavailableError(f"Could not initialize V3 model: {exc}") from exc


def preload_models(modalities: list[str] | None = None) -> None:
    """Safely preloads models into memory so user requests do not experience cold-start latency."""
    if modalities is None:
        modalities = ["image", "audio", "video"]
    for mod in modalities:
        try:
            _load_model(mod)
        except Exception as exc:
            logger.debug("Preloading %s model skipped or failed: %s", mod, exc)
    try:
        _load_v3_image_model()
    except Exception as exc:
        logger.debug("Preloading V3 image model skipped or failed: %s", exc)


def route_image(file_path: str | Path) -> dict:
    """
    Resolution and quality-gated routing logic:
    - High-resolution, uncompressed photographic images: Model V3
    - Low-resolution crops, extreme aspect ratios, or heavily compressed files: Model V2
    - If V3 checkpoint is unavailable or corrupt: Conservative fallback to Model V2
    """
    file_path = Path(file_path)
    v3_path = MODEL_DIR / "image" / "best_model_v3.keras"
    v3_available = v3_path.is_file() and v3_path.stat().st_size > 0

    if not v3_available:
        return {
            "route": "v2",
            "model_version": "V2",
            "model_name": "Model V2 (MobileNetV2 Standard CNN)",
            "threshold": get_threshold("image", "v2"),
            "reason": "Model V3 is not available on disk. Fallback to Model V2.",
        }

    try:
        file_size = file_path.stat().st_size
        with Image.open(file_path) as im:
            w, h = im.size
            fmt = (im.format or "").upper()
    except Exception as exc:
        return {
            "route": "v2",
            "model_version": "V2",
            "model_name": "Model V2 (MobileNetV2 Standard CNN)",
            "threshold": get_threshold("image", "v2"),
            "reason": f"Could not decode image metadata ({exc}). Conservative fallback to Model V2.",
        }

    min_dim = min(w, h)
    max_dim = max(w, h)

    # 1. Dimension Check:
    # Requires max_dim >= 512 and min_dim >= 320 to provide valid optical detail
    if max_dim < 512 or min_dim < 320:
        return {
            "route": "v2",
            "model_version": "V2",
            "model_name": "Model V2 (MobileNetV2 Standard CNN)",
            "threshold": get_threshold("image", "v2"),
            "reason": f"Low resolution ({w}x{h}, max={max_dim}<512 or min={min_dim}<320). Routed to Model V2.",
            "width": w, "height": h, "file_size": file_size, "format": fmt,
        }

    # 2. Aspect Ratio Check:
    ratio = max_dim / max(1, min_dim)
    if ratio > 3.0:
        return {
            "route": "v2",
            "model_version": "V2",
            "model_name": "Model V2 (MobileNetV2 Standard CNN)",
            "threshold": get_threshold("image", "v2"),
            "reason": f"Extreme aspect ratio ({ratio:.2f} > 3.0). Conservative fallback to Model V2.",
            "width": w, "height": h, "file_size": file_size, "format": fmt,
        }

    # 3. Compression Indicator (Bytes Per Pixel on lossy JPEG):
    bpp = file_size / (w * h)
    if fmt in ["JPEG", "JPG"] and bpp < 0.04:
        return {
            "route": "v2",
            "model_version": "V2",
            "model_name": "Model V2 (MobileNetV2 Standard CNN)",
            "threshold": get_threshold("image", "v2"),
            "reason": f"Heavy lossy compression detected (BPP {bpp:.4f} < 0.04). PRNU degraded; routed to Model V2.",
            "width": w, "height": h, "file_size": file_size, "format": fmt,
        }

    # 4. Route to V3
    return {
        "route": "v3",
        "model_version": "V3",
        "model_name": "Model V3 (Dual-Branch Spatial-Frequency)",
        "threshold": get_threshold("image", "v3"),
        "reason": f"High resolution ({w}x{h}), normal aspect ratio ({ratio:.2f}), adequate quality (BPP {bpp:.3f}). Routed to Model V3.",
        "width": w, "height": h, "file_size": file_size, "format": fmt,
    }


def _preprocess_image_v2(file_path: str | Path) -> np.ndarray:
    """Standard V2 preprocessing: center-crop square and bilinear resize to 224x224 with MobileNetV2 normalization."""
    file_path = Path(file_path)
    try:
        with Image.open(file_path) as img:
            im = img.convert("RGB")
            w, h = im.size
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            im_cropped = im.crop((left, top, left + min_dim, top + min_dim))
            im_resized = im_cropped.resize((224, 224), Image.Resampling.BILINEAR)
            arr = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(im_resized, dtype=np.float32))
    except Exception as exc:
        raise MediaDecodeError("The uploaded image is invalid or could not be decoded.") from exc
    return np.expand_dims(arr, axis=0)


# Backward-compatible alias for existing tests and external callers
_preprocess_image = _preprocess_image_v2


def _preprocess_image_v3(file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """V3 preprocessing: dual global 224x224 view and focal center 60% crop view."""
    file_path = Path(file_path)
    try:
        with Image.open(file_path) as img:
            im = img.convert("RGB")
            w, h = im.size
            # Global view
            im_global = im.resize((224, 224), Image.Resampling.BILINEAR)
            arr_global = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(im_global, dtype=np.float32))

            # Focal view: central 60% crop
            crop_w, crop_h = int(w * 0.60), int(h * 0.60)
            left = (w - crop_w) // 2
            top = (h - crop_h) // 2
            focal_crop = im.crop((left, top, left + crop_w, top + crop_h))
            focal_resized = focal_crop.resize((224, 224), Image.Resampling.BILINEAR)
            arr_focal = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(focal_resized, dtype=np.float32))
    except Exception as exc:
        raise MediaDecodeError("The uploaded image is invalid or could not be decoded.") from exc
    return np.expand_dims(arr_global, axis=0), np.expand_dims(arr_focal, axis=0)


def predict_image(file_path: str | Path) -> dict:
    """
    Runs the Resolution-Gated Dual Pipeline for images.
    - Decides route based on resolution and quality indicators.
    - Routes to V3 for high-res modern imagery, with automatic graceful fallback to V2.
    - Routes to V2 for legacy/low-res/compressed images.
    """
    file_path = Path(file_path)
    route_info = route_image(file_path)
    selected_route = route_info["route"]
    model_version = route_info["model_version"]
    model_name = route_info["model_name"]
    threshold = route_info["threshold"]
    routing_reason = route_info["reason"]

    if selected_route == "v3":
        try:
            model_v3 = _load_v3_image_model()
            arr_g, arr_f = _preprocess_image_v3(file_path)
            raw_output = np.asarray(model_v3.predict([arr_g, arr_f], verbose=0), dtype=np.float32).reshape(-1)
        except Exception as exc:
            logger.warning("V3 inference failed (%s); triggering conservative fallback to Model V2", exc)
            model_version = "V2 (fallback)"
            model_name = "Model V2 (Fallback from V3)"
            threshold = get_threshold("image", "v2")
            routing_reason = f"V3 failed: {exc}. Executed conservative fallback to V2."
            model_v2 = _load_model("image")
            model_input = _preprocess_image(file_path)
            raw_output = np.asarray(model_v2.predict(model_input, verbose=0), dtype=np.float32).reshape(-1)
    else:
        model_v2 = _load_model("image")
        model_input = _preprocess_image(file_path)
        raw_output = np.asarray(model_v2.predict(model_input, verbose=0), dtype=np.float32).reshape(-1)

    if (
        raw_output.size != 1
        or not np.all(np.isfinite(raw_output))
        or np.any((raw_output < 0.0) | (raw_output > 1.0))
    ):
        raise PredictionError("The model returned an invalid prediction score.")

    probability_real = float(raw_output[0])
    calib = get_calibration_details("image", model_version=model_version)
    return compute_prediction_result(
        probability_real=probability_real,
        threshold=threshold,
        uncertainty_lower=calib["uncertainty_lower"],
        uncertainty_upper=calib["uncertainty_upper"],
        modality="image",
        extra_fields={
            "model_used": model_name,
            "model_version": model_version,
            "routing_reason": routing_reason,
        },
    )


def _preprocess_video_windows(file_path: Path, num_windows: int = 3, window_size: int = 5) -> np.ndarray:
    file_path = Path(file_path)
    capture = cv2.VideoCapture(str(file_path))
    if not capture.isOpened():
        capture.release()
        raise MediaDecodeError("The uploaded video could not be opened. Check its codec and file integrity.")
    try:
        success, first_frame = capture.read()
        if not success or first_frame is None:
            raise MediaDecodeError("The uploaded video contains no readable frames.")
    finally:
        capture.release()

    try:
        windows = extract_video_windows(
            str(file_path),
            num_windows=num_windows,
            window_size=window_size,
            image_size=(224, 224),
            strict=True,
        )
    except Exception as exc:
        raise MediaDecodeError("The uploaded video could not be decoded.") from exc
    if len(windows.shape) != 5 or windows.shape[1:] != (window_size, 224, 224, 3):
        raise MediaDecodeError("The uploaded video could not be converted to model input.")
    return windows


def _preprocess_video(file_path: Path, num_frames: int = 5) -> np.ndarray:
    file_path = Path(file_path)
    capture = cv2.VideoCapture(str(file_path))
    if not capture.isOpened():
        capture.release()
        raise MediaDecodeError("The uploaded video could not be opened. Check its codec and file integrity.")
    try:
        success, first_frame = capture.read()
        if not success or first_frame is None:
            raise MediaDecodeError("The uploaded video contains no readable frames.")
    finally:
        capture.release()

    try:
        frames = extract_frames_from_video(
            str(file_path),
            num_frames=num_frames,
            image_size=(224, 224),
            strict=True,
        )
    except Exception as exc:
        raise MediaDecodeError("The uploaded video could not be decoded.") from exc
    if frames.shape != (num_frames, 224, 224, 3):
        raise MediaDecodeError("The uploaded video could not be converted to model input.")
    return np.expand_dims(frames, axis=0)


def predict_video(file_path: str | Path, num_frames: int = 5) -> dict:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise MediaDecodeError("The uploaded file could not be found.")
    detect_modality(file_path, "video")

    windows = _preprocess_video_windows(file_path, num_windows=3, window_size=5)
    model = _load_model("video")
    try:
        window_predictions = np.asarray(
            model.predict(windows, verbose=0),
            dtype=np.float32,
        ).reshape(-1)
    except Exception as exc:
        raise PredictionError("The model could not analyze this video. Please try another upload.") from exc

    if (
        window_predictions.size == 0
        or not np.all(np.isfinite(window_predictions))
        or np.any((window_predictions < 0.0) | (window_predictions > 1.0))
    ):
        raise PredictionError("The model returned an invalid video prediction score.")

    if len(window_predictions) > 1:
        probability_real = float(0.35 * np.min(window_predictions) + 0.65 * np.mean(window_predictions))
    else:
        probability_real = float(window_predictions[0])

    total_frames_analyzed = int(windows.shape[0] * windows.shape[1])
    calib = get_calibration_details("video")
    return compute_prediction_result(
        probability_real=probability_real,
        threshold=calib["threshold"],
        uncertainty_lower=calib["uncertainty_lower"],
        uncertainty_upper=calib["uncertainty_upper"],
        modality="video",
        extra_fields={
            "frames_analyzed": total_frames_analyzed,
            "windows_analyzed": int(windows.shape[0]),
            "model_used": "Video Deepfake Detector V2 (MobileNetV2 + Bi-GRU)",
            "model_version": "V2",
        },
    )


def _preprocess_audio(file_path: str | Path) -> np.ndarray:
    file_path = Path(file_path)
    try:
        features = extract_audio_features(str(file_path), sr=16000, duration=3.0, strict=True)
    except Exception as exc:
        msg = str(exc)
        if "could not be decoded" not in msg.lower():
            msg = f"The uploaded audio could not be decoded. {msg}"
        raise MediaDecodeError(msg) from exc
    return np.expand_dims(features, axis=0)


def predict_audio(file_path: str | Path) -> dict:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise MediaDecodeError("The uploaded file could not be found.")
    detect_modality(file_path, "audio")

    model_input = _preprocess_audio(file_path)
    model = _load_model("audio")
    try:
        model_output = np.asarray(model.predict(model_input, verbose=0), dtype=np.float32).reshape(-1)
    except Exception as exc:
        raise PredictionError("The model could not analyze this file. Please try another upload.") from exc
    if (
        model_output.size != 1
        or not np.all(np.isfinite(model_output))
        or np.any((model_output < 0.0) | (model_output > 1.0))
    ):
        raise PredictionError("The model returned an invalid prediction score.")

    probability_real = float(model_output[0])
    calib = get_calibration_details("audio")
    return compute_prediction_result(
        probability_real=probability_real,
        threshold=calib["threshold"],
        uncertainty_lower=calib["uncertainty_lower"],
        uncertainty_upper=calib["uncertainty_upper"],
        modality="audio",
        extra_fields={
            "model_used": "Audio Deepfake Detector V2 (Acoustic Feature MLP)",
            "model_version": "V2",
        },
    )


def predict_file(file_path: str | Path, modality: str | None = None) -> dict:
    file_path = Path(file_path)
    if not file_path.is_file():
        raise MediaDecodeError("The uploaded file could not be found.")
    modality = detect_modality(file_path, modality)

    if modality == "image":
        return predict_image(file_path)

    if modality == "video":
        return predict_video(file_path)

    if modality == "audio":
        return predict_audio(file_path)

    raise UnsupportedMediaError(f"Modality '{modality}' is not supported.")


def predict_single_file(file_path: str) -> dict:
    return predict_file(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict deepfake status for a media file")
    parser.add_argument("file_path", help="Path to an audio, image, or video file")
    args = parser.parse_args()
    try:
        result = predict_single_file(args.file_path)
        print(result)
    except PredictionError as exc:
        parser.exit(2, f"Prediction failed: {exc}\n")
