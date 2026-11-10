from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import predict


@pytest.mark.parametrize(
    ("probability_real", "expected_label", "expected_confidence"),
    [
        (0.339, "Fake", 0.661),
        (0.34, "Real", 0.34),
        (0.4, "Real", 0.4),
    ],
)
def test_image_confidence_matches_threshold_selected_class(
    tmp_path,
    probability_real,
    expected_label,
    expected_confidence,
):
    image_path = tmp_path / "sample.png"
    image_path.touch()
    model = MagicMock()
    model.predict.return_value = np.array([[probability_real]], dtype=np.float32)

    with (
        patch.object(predict, "_preprocess_image", return_value=np.zeros((1, 224, 224, 3))),
        patch.object(predict, "_load_model", return_value=model) as load_model,
        patch.object(predict, "get_threshold", return_value=0.34),
    ):
        result = predict.predict_file(image_path, "image")

    load_model.assert_called_once_with("image")
    assert result["prediction"] == expected_label
    assert result["confidence"] == pytest.approx(expected_confidence)


def test_image_rejects_non_scalar_model_output(tmp_path):
    image_path = tmp_path / "sample.png"
    image_path.touch()
    model = MagicMock()
    model.predict.return_value = np.array([[0.2, 0.8]], dtype=np.float32)

    with (
        patch.object(predict, "_preprocess_image", return_value=np.zeros((1, 224, 224, 3))),
        patch.object(predict, "_load_model", return_value=model),
        pytest.raises(predict.PredictionError, match="invalid prediction score"),
    ):
        predict.predict_file(image_path, "image")


def test_corrupt_image_is_reported_as_a_decode_error(tmp_path):
    image_path = tmp_path / "corrupt.png"
    image_path.write_bytes(b"not an image")

    with pytest.raises(predict.MediaDecodeError, match="invalid or could not be decoded"):
        predict._preprocess_image(image_path)
