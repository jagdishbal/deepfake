from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import predict
from preprocessing.clean_video import extract_frames_from_video


def test_video_threshold_confidence_uses_video_model_and_sequence(tmp_path):
    video_path = tmp_path / "sample.mp4"
    video_path.touch()
    frames = np.zeros((1, 10, 224, 224, 3), dtype=np.float32)
    model = MagicMock()
    model.predict.return_value = np.array([[0.52]], dtype=np.float32)

    with (
        patch.object(predict, "_preprocess_video", return_value=frames),
        patch.object(predict, "_load_model", return_value=model) as load_model,
        patch.object(predict, "get_threshold", return_value=0.54),
    ):
        result = predict.predict_video(video_path)

    load_model.assert_called_once_with("video")
    assert model.predict.call_args.args[0].shape == (1, 10, 224, 224, 3)
    assert result["prediction"] == "Fake"
    assert result["confidence"] == pytest.approx(0.48)
    assert result["frames_analyzed"] == 10


def test_strict_video_frame_decode_rejects_unreadable_sample(tmp_path):
    video_path = tmp_path / "corrupt.mp4"
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.get.return_value = 10
    capture.read.return_value = (False, None)

    with (
        patch("preprocessing.clean_video.cv2.VideoCapture", return_value=capture),
        pytest.raises(ValueError, match="sampled video frame"),
    ):
        extract_frames_from_video(str(video_path), num_frames=10, strict=True)

    capture.release.assert_called_once()


def test_video_frames_use_mobilenet_v2_input_range(tmp_path):
    video_path = tmp_path / "sample.mp4"
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.get.return_value = 1
    capture.read.return_value = (True, np.full((2, 2, 3), 127, dtype=np.uint8))

    with patch("preprocessing.clean_video.cv2.VideoCapture", return_value=capture):
        frames = extract_frames_from_video(
            str(video_path),
            num_frames=1,
            image_size=(1, 1),
        )

    assert frames.shape == (1, 1, 1, 3)
    np.testing.assert_allclose(frames, 127 / 127.5 - 1.0, atol=1e-6)
