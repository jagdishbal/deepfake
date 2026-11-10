import os
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
import numpy as np
import tensorflow as tf
import librosa
import soundfile as sf
from imageio_ffmpeg import get_ffmpeg_exe

MAX_AUDIO_DURATION_SECONDS = 300.0  # 5 minutes maximum allowed audio duration


def probe_audio(file_path: str | Path) -> tuple[bool, str | None, float | None, str]:
    """
    Probes media file using FFmpeg to check for audio streams, codec, and duration.
    Returns (has_audio_stream, codec_name, duration_in_seconds, stderr).
    """
    file_path = Path(file_path)
    if not file_path.is_file() or file_path.stat().st_size == 0:
        return False, None, 0.0, ""

    ffmpeg_exe = get_ffmpeg_exe()
    cmd = [ffmpeg_exe, "-nostdin", "-i", str(file_path)]
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="ignore")
    stderr = proc.stderr

    # Check for audio stream: e.g. "Stream #0:1: Audio: aac" or "Stream #0:0: Audio: mp3"
    audio_match = re.search(r"Stream\s+#\d+:\d+.*Audio:\s+([a-zA-Z0-9_-]+)", stderr, re.IGNORECASE)
    has_audio = bool(audio_match)
    codec = audio_match.group(1).lower() if audio_match else None

    # Check for duration: "Duration: 00:03:15.42"
    duration = None
    dur_match = re.search(r"Duration:\s+(\d+):(\d+):(\d+\.?\d*)", stderr)
    if dur_match:
        h, m, s = dur_match.groups()
        duration = int(h) * 3600 + int(m) * 60 + float(s)

    return has_audio, codec, duration, stderr


def extract_audio_track_ffmpeg(input_path: str | Path, target_sr: int = 16000) -> Path:
    """
    Extracts/decodes audio from any supported container (MP3, MP4, AAC, M4A, etc.)
    into a standardized mono 16-bit PCM WAV at target_sr (16000 Hz).
    Returns path to a temporary WAV file in the system temp directory.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f"The audio file could not be found: {input_path}")
    if input_path.stat().st_size == 0:
        raise ValueError("The uploaded audio file is empty.")

    ext = input_path.suffix.lower()

    # Probe file for streams, codec, and duration
    has_audio, codec, duration, probe_stderr = probe_audio(input_path)
    if not has_audio:
        if ext == ".mp4":
            raise ValueError("The uploaded MP4 file does not contain an audio track. Please upload media with audio.")
        raise ValueError("The uploaded file does not contain an audio track. Please upload media with audio.")

    # Format-specific validation
    if ext in {".mp3"}:
        if not codec or not ("mp3" in codec or "mpeg" in codec):
            raise ValueError("The uploaded audio could not be decoded. Check its format and codec.")
        if "corrupt bitstream" in probe_stderr.lower() or "error submitting packet" in probe_stderr.lower():
            raise ValueError("The uploaded audio could not be decoded. Check its format and codec.")

    if duration is not None:
        if duration <= 0.0:
            raise ValueError("The uploaded audio file contains no playable audio.")
        if duration > MAX_AUDIO_DURATION_SECONDS:
            raise ValueError(
                f"The audio duration ({duration:.1f}s) exceeds the allowed limit of {int(MAX_AUDIO_DURATION_SECONDS)} seconds (5 minutes)."
            )

    ffmpeg_exe = get_ffmpeg_exe()
    temp_dir = Path(tempfile.gettempdir())
    temp_wav = temp_dir / f"extracted_{uuid.uuid4().hex}.wav"

    cmd = [
        ffmpeg_exe,
        "-nostdin",
        "-y",
        "-i", str(input_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(target_sr),
        "-ac", "1",
        str(temp_wav),
    ]

    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, errors="ignore")
    if proc.returncode != 0 or not temp_wav.is_file() or temp_wav.stat().st_size == 0:
        temp_wav.unlink(missing_ok=True)
        if "does not contain any stream" in proc.stderr or "matches no streams" in proc.stderr:
            raise ValueError("The uploaded MP4 file does not contain an audio track. Please upload media with audio.")
        raise ValueError("The uploaded audio could not be decoded. Check its format and codec.")

    # Check if decoder output indicates a fatal corruption
    if "error submitting packet" in proc.stderr.lower() or "corrupt bitstream" in proc.stderr.lower():
        temp_wav.unlink(missing_ok=True)
        raise ValueError("The uploaded audio could not be decoded. Check its format and codec.")

    return temp_wav


def extract_audio_features(audio_path: str | Path, sr: int = 16000, duration: float = 3.0, strict: bool = False):
    """
    Loads audio, resamples, pads/truncates, and computes 140 acoustic features.
    Features:
    - MFCCs (20 coefficients): mean (20), std (20) -> 40
    - Delta MFCCs: mean (20), std (20) -> 40
    - Delta-2 MFCCs: mean (20), std (20) -> 40
    - Spectral Contrast: mean (7), std (7) -> 14
    - Spectral Rolloff: mean (1), std (1) -> 2
    - Spectral Flatness: mean (1), std (1) -> 2
    - Zero Crossing Rate: mean (1), std (1) -> 2
    Total: 140 features.
    Supports WAV, MP3, MP4, AAC, M4A, FLAC, OGG.
    - WAV: Existing processing via librosa.load, with FFmpeg fallback if needed.
    - MP3: Decoded and converted to 16kHz mono PCM via FFmpeg.
    - MP4: Audio track extracted and converted to 16kHz mono PCM via FFmpeg.
    """
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        if strict:
            raise ValueError(f"The audio file could not be found: {audio_path}")
        return np.zeros(140, dtype=np.float32)
    if audio_path.stat().st_size == 0:
        if strict:
            raise ValueError("The uploaded audio file is empty.")
        return np.zeros(140, dtype=np.float32)

    ext = audio_path.suffix.lower()
    temp_wav = None

    try:
        if ext == ".wav":
            # Existing processing for WAV
            try:
                info = sf.info(str(audio_path))
                if info.duration <= 0:
                    raise ValueError("The uploaded audio file contains no playable audio.")
                if info.duration > MAX_AUDIO_DURATION_SECONDS:
                    raise ValueError(
                        f"The audio duration ({info.duration:.1f}s) exceeds the allowed limit of {int(MAX_AUDIO_DURATION_SECONDS)} seconds (5 minutes)."
                    )
                y, _ = librosa.load(str(audio_path), sr=sr, duration=duration)
            except ValueError:
                raise
            except Exception:
                # If librosa fails on non-standard WAV encoding, fallback to FFmpeg extraction
                temp_wav = extract_audio_track_ffmpeg(audio_path, target_sr=sr)
                y, _ = sf.read(str(temp_wav), dtype="float32")
        elif ext in {".mp3", ".mp4", ".aac", ".m4a", ".flac", ".ogg", ".webm", ".avi", ".mov"}:
            # Transcode/extract audio to 16kHz mono PCM WAV via FFmpeg
            temp_wav = extract_audio_track_ffmpeg(audio_path, target_sr=sr)
            y, _ = sf.read(str(temp_wav), dtype="float32")
        else:
            raise ValueError(f"Unsupported audio format: {ext}")

    except Exception as exc:
        if temp_wav is not None:
            try:
                temp_wav.unlink(missing_ok=True)
            except Exception:
                pass
        if strict:
            if isinstance(exc, ValueError):
                raise
            raise ValueError(f"Unable to decode audio file: {audio_path}") from exc
        # Return zeros if corrupted and non-strict
        return np.zeros(140, dtype=np.float32)
    finally:
        if temp_wav is not None:
            try:
                temp_wav.unlink(missing_ok=True)
            except Exception:
                pass

    # Pad if too short
    target_len = int(sr * duration)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)))
    else:
        y = y[:target_len]

    # Compute 140 acoustic features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    flatness = librosa.feature.spectral_flatness(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)

    feats = np.hstack([
        np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
        np.mean(delta, axis=1), np.std(delta, axis=1),
        np.mean(delta2, axis=1), np.std(delta2, axis=1),
        np.mean(contrast, axis=1), np.std(contrast, axis=1),
        np.mean(rolloff), np.std(rolloff),
        np.mean(flatness), np.std(flatness),
        np.mean(zcr), np.std(zcr),
    ])
    return feats.astype(np.float32)


def audio_generator(directory: str, sr: int = 16000, duration: float = 3.0):
    """Generator yielding audio acoustic features and label."""
    classes = ['fake', 'real']
    for label, class_name in enumerate(classes):
        class_dir = Path(directory) / class_name
        if not class_dir.exists():
            continue

        for audio_path in class_dir.rglob("*"):
            if not audio_path.is_file():
                continue

            ext = audio_path.suffix.lower()
            if ext not in {".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a", ".mp4"}:
                continue

            features = extract_audio_features(str(audio_path), sr, duration)
            # Binary label: fake=0, real=1
            yield features, np.array([label], dtype=np.float32)


def get_audio_dataset(directory: str | Path, batch_size: int = 32, sr: int = 16000, duration: float = 3.0, shuffle: bool = True):
    """
    Creates a tf.data.Dataset for audio, computing acoustic features on the fly.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory {dir_path} does not exist.")

    dataset = tf.data.Dataset.from_generator(
        lambda: audio_generator(str(dir_path), sr, duration),
        output_signature=(
            tf.TensorSpec(shape=(140,), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    )

    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
