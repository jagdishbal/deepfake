from io import BytesIO
from unittest.mock import patch

import app as app_module
import pytest
from werkzeug.datastructures import FileStorage


class FakeCursor:
    def __init__(self, row):
        self.row = row
        self.query_parameters = None

    def execute(self, query, parameters):
        self.query_parameters = parameters

    def fetchone(self):
        return self.row

    def close(self):
        pass


class FakeConnection:
    def __init__(self, row):
        self.cursor_instance = FakeCursor(row)

    def cursor(self, dictionary=False):
        return self.cursor_instance

    def close(self):
        pass


def test_empty_upload_is_rejected_and_removed(tmp_path):
    upload = FileStorage(stream=BytesIO(b""), filename="empty.png")

    with (
        patch.object(app_module, "UPLOAD_FOLDER", tmp_path),
        pytest.raises(app_module.UploadValidationError, match="empty"),
    ):
        app_module._save_upload(upload, "image")

    assert list(tmp_path.iterdir()) == []


def test_upload_redirects_to_its_saved_result_and_displays_that_record(tmp_path):
    app_module.app.config.update(TESTING=True)
    uploaded_path = tmp_path / "stored.png"
    uploaded_path.touch()
    stored_result = {
        "id": 42,
        "prediction": "Real",
        "filename": "stored.png",
        "file_type": "Image",
        "confidence": 42.5,
    }
    connection = FakeConnection(stored_result)

    with (
        patch.object(app_module, "_save_upload", return_value=("stored.png", uploaded_path)),
        patch.object(
            app_module,
            "predict_media",
            return_value={"prediction": "Real", "confidence": 0.425},
        ),
        patch.object(app_module, "_record_prediction", return_value=42),
        app_module.app.test_client() as client,
    ):
        with client.session_transaction() as session:
            session["user_id"] = 7

        response = client.post(
            "/image_detection",
            data={"image": (BytesIO(b"image bytes"), "sample.png")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 302
        assert response.headers["Location"].endswith("/result/42")

        with patch.object(app_module, "get_db_connection", return_value=connection):
            result_response = client.get(response.headers["Location"])

    assert connection.cursor_instance.query_parameters == (42, 7)
    assert result_response.status_code == 200
    assert b"stored.png" in result_response.data
    assert b"42.50%" in result_response.data
    assert b"Uncalibrated model score" in result_response.data
    assert b"not a calibrated probability" in result_response.data
