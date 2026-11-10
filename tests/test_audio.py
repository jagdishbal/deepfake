from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import predict


def test_audio_uses_audio_features_model_and_threshold_class_confidence(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.touch()
    features = np.zeros((1, 128, 94, 1), dtype=np.float32)
    model = MagicMock()
    model.predict.return_value = np.array([[0.49]], dtype=np.float32)

    with (
        patch.object(predict, "_preprocess_audio", return_value=features),
        patch.object(predict, "_load_model", return_value=model) as load_model,
        patch.object(predict, "get_threshold", return_value=0.48),
    ):
        result = predict.predict_file(audio_path, "audio")

    load_model.assert_called_once_with("audio")
    model.predict.assert_called_once_with(features, verbose=0)
    assert result["modality"] == "audio"
    assert result["prediction"] == "Real"
    assert result["confidence"] == pytest.approx(0.49)


def test_audio_decode_failure_is_not_converted_to_a_prediction(tmp_path):
    audio_path = tmp_path / "corrupt.wav"
    audio_path.touch()

    with (
        patch.object(predict, "extract_audio_features", side_effect=ValueError("decode failed")),
        pytest.raises(predict.MediaDecodeError, match="could not be decoded"),
    ):
        predict._preprocess_audio(audio_path)
