"""
Unit and integration tests for corrected audio and video deepfake detection pipelines.
Verifies proper dual probability calculation, inconclusive reporting, feature shapes,
and corruption/error handling.
"""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from pathlib import Path

import predict
from preprocessing.clean_audio import extract_audio_features
from preprocessing.clean_video import extract_frames_from_video


def test_audio_feature_extraction_dimension():
    """Verify that extract_audio_features returns exact 140 acoustic features."""
    # Test fallback zeros when audio is not present or non-strict
    feats = extract_audio_features("non_existent_file.wav", strict=False)
    assert isinstance(feats, np.ndarray)
    assert feats.shape == (140,)
    assert feats.dtype == np.float32


def test_confidence_and_dual_probabilities():
    """Verify that compute_prediction_result returns accurate dual probabilities and confidence."""
    # High confidence Real: probability_real = 0.92
    res_real = predict.compute_prediction_result(
        probability_real=0.92,
        threshold=0.30,
        uncertainty_lower=0.22,
        uncertainty_upper=0.38,
        modality="audio",
    )
    assert res_real["prediction"] == "Real"
    assert pytest.approx(res_real["probability_real"], 0.01) == 0.92
    assert pytest.approx(res_real["probability_fake"], 0.01) == 0.08
    assert pytest.approx(res_real["confidence"], 0.01) == 0.92
    assert res_real["is_inconclusive"] is False
    assert res_real["confidence_status"] == "CONFIDENT"
    assert "REAL: 92.0%" in res_real["dual_probabilities_text"]
    assert "FAKE: 8.0%" in res_real["dual_probabilities_text"]

    # High confidence Fake: probability_real = 0.05
    res_fake = predict.compute_prediction_result(
        probability_real=0.05,
        threshold=0.30,
        uncertainty_lower=0.22,
        uncertainty_upper=0.38,
        modality="audio",
    )
    assert res_fake["prediction"] == "Fake"
    assert pytest.approx(res_fake["probability_real"], 0.01) == 0.05
    assert pytest.approx(res_fake["probability_fake"], 0.01) == 0.95
    assert pytest.approx(res_fake["confidence"], 0.01) == 0.95
    assert res_fake["is_inconclusive"] is False
    assert res_fake["confidence_status"] == "CONFIDENT"
    assert "REAL: 5.0%" in res_fake["dual_probabilities_text"]
    assert "FAKE: 95.0%" in res_fake["dual_probabilities_text"]


def test_inconclusive_flag_near_decision_boundary():
    """Verify that probabilities near the threshold are marked INCONCLUSIVE / LOW CONFIDENCE."""
    # Near threshold 0.30: probability_real = 0.28 (within [0.22, 0.38])
    res = predict.compute_prediction_result(
        probability_real=0.28,
        threshold=0.30,
        uncertainty_lower=0.22,
        uncertainty_upper=0.38,
        modality="video",
    )
    assert res["is_inconclusive"] is True
    assert res["confidence_status"] == "INCONCLUSIVE / LOW CONFIDENCE"

    # Also test confidence < 0.60
    res_low = predict.compute_prediction_result(
        probability_real=0.55,
        threshold=0.50,
        uncertainty_lower=0.48,
        uncertainty_upper=0.52,
        modality="image",
    )
    assert res_low["confidence"] == pytest.approx(0.55)
    assert res_low["is_inconclusive"] is True
    assert res_low["confidence_status"] == "INCONCLUSIVE / LOW CONFIDENCE"


def test_video_prediction_dynamic_frames(tmp_path):
    """Verify video prediction supports variable sequence lengths (e.g. 5 or 10 frames)."""
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()

    for frames_count in [5, 10]:
        frames = np.zeros((1, frames_count, 224, 224, 3), dtype=np.float32)
        model = MagicMock()
        model.predict.return_value = np.array([[0.85]], dtype=np.float32)

        with (
            patch.object(predict, "_preprocess_video", return_value=frames),
            patch.object(predict, "_load_model", return_value=model),
            patch.object(predict, "get_calibration_details", return_value={
                "threshold": 0.30, "uncertainty_lower": 0.22, "uncertainty_upper": 0.38
            }),
        ):
            res = predict.predict_video(video_path, num_frames=frames_count)

        assert res["prediction"] == "Real"
        assert res["frames_analyzed"] == frames_count
        assert pytest.approx(res["probability_real"], 0.01) == 0.85
        assert pytest.approx(res["probability_fake"], 0.01) == 0.15
        assert res["is_inconclusive"] is False


def test_corrupted_audio_file_raises_decode_error(tmp_path):
    """Verify that corrupted or unreadable audio fails with MediaDecodeError."""
    bad_audio = tmp_path / "corrupt.mp3"
    bad_audio.write_bytes(b"NOT_A_REAL_MP3_STREAM")

    with pytest.raises(predict.MediaDecodeError):
        predict._preprocess_audio(bad_audio)
