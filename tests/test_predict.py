from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

import predict


def test_missing_threshold_configuration_fails_instead_of_using_a_default():
    with (
        patch("builtins.open", side_effect=OSError("configuration missing")),
        pytest.raises(predict.PredictionError, match="threshold configuration"),
    ):
        predict.get_threshold("image")


def test_missing_checkpoint_fails_explicitly(tmp_path):
    predict._load_model.cache_clear()
    try:
        with (
            patch.object(predict, "MODEL_DIR", tmp_path),
            pytest.raises(predict.ModelUnavailableError, match="not available"),
        ):
            predict._load_model("image")
    finally:
        predict._load_model.cache_clear()


def test_unsupported_extension_is_rejected(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.touch()

    with pytest.raises(predict.UnsupportedMediaError, match="not supported"):
        predict.predict_file(file_path, "image")


def test_image_preprocessing_matches_training_normalization(tmp_path):
    image_path = tmp_path / "mid_gray.png"
    Image.new("RGB", (2, 2), color=(128, 128, 128)).save(image_path)

    model_input = predict._preprocess_image(image_path)

    assert model_input.shape == (1, 224, 224, 3)
    np.testing.assert_allclose(model_input, 128 / 127.5 - 1.0, atol=1e-6)


def test_video_prediction_uses_video_model_and_video_batch_shape(tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.touch()
    frames = np.zeros((10, 224, 224, 3), dtype=np.float32)
    batched_frames = np.expand_dims(frames, axis=0)
    model = MagicMock()
    model.predict.return_value = np.array([[0.7]], dtype=np.float32)

    with (
        patch.object(predict, "_preprocess_video", return_value=batched_frames),
        patch.object(predict, "_load_model", return_value=model) as load_model,
        patch.object(predict, "get_threshold", return_value=0.54),
    ):
        result = predict.predict_video(video_path)

    load_model.assert_called_once_with("video")
    model.predict.assert_called_once()
    model_input = model.predict.call_args.args[0]
    assert model_input.shape == (1, 10, 224, 224, 3)
    assert result["prediction"] == "Real"
    assert np.isclose(result["confidence"], 0.7)
    assert np.isclose(result["probability_real"], 0.7)
    assert result["modality"] == "video"
    assert result["frames_analyzed"] == 10
