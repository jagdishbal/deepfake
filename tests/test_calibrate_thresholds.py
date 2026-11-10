import json

from calibrate_thresholds import save_calibration


def test_save_calibration_updates_one_modality_and_preserves_others(tmp_path):
    output_path = tmp_path / "calibration.json"
    output_path.write_text(
        json.dumps({"image": {"threshold": 0.4}, "audio": {"threshold": 0.5}}),
        encoding="utf-8",
    )

    save_calibration({"image": {"threshold": 0.6}}, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == {
        "image": {"threshold": 0.6},
        "audio": {"threshold": 0.5},
    }


def test_save_calibration_rejects_non_object_configuration(tmp_path):
    output_path = tmp_path / "calibration.json"
    output_path.write_text("[]", encoding="utf-8")

    try:
        save_calibration({"image": {"threshold": 0.6}}, output_path)
    except ValueError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("Expected invalid calibration configuration to be rejected")
