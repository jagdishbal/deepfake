import hashlib
import io
import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import numpy as np
import tensorflow as tf
from app import app
from database.database import get_db_connection
from imageio_ffmpeg import get_ffmpeg_exe
from predict import (
    SUPPORTED_EXTENSIONS,
    _load_model,
    _load_v3_image_model,
    _preprocess_audio,
    _preprocess_image_v2,
    _preprocess_image_v3,
    _preprocess_video,
    detect_modality,
    get_threshold,
    predict_file,
    route_image,
)
from preprocessing.clean_audio import (
    MAX_AUDIO_DURATION_SECONDS,
    extract_audio_features,
    extract_audio_track_ffmpeg,
    probe_audio,
)
from services.audit_service import get_prediction_audit

ffmpeg = get_ffmpeg_exe()
base_dir = Path(config.BASE_DIR)
fixtures_dir = base_dir / "runtime" / "temp" / "audio_fixtures"
fixtures_dir.mkdir(parents=True, exist_ok=True)

report_data = {}

def run_phase1_and_phase2():
    print("=" * 80)
    print("PHASE 1 & 2: AUDIO FORMAT & PREDICTION VERIFICATION")
    print("=" * 80)
    
    # Base real WAV from dataset
    base_wav = base_dir / "dataset" / "audio" / "test" / "fake" / "00001_id01683_wavtolip.wav"
    assert base_wav.is_file(), f"Base sample wav missing: {base_wav}"
    
    # 1. WAV
    wav_path = fixtures_dir / "valid_sample.wav"
    wav_path.write_bytes(base_wav.read_bytes())
    
    # 2. MP3
    mp3_path = fixtures_dir / "valid_sample.mp3"
    subprocess.run([ffmpeg, "-nostdin", "-y", "-i", str(base_wav), "-acodec", "libmp3lame", "-b:a", "128k", str(mp3_path)], stdin=subprocess.DEVNULL, capture_output=True, check=True)
    
    # 3. MP4 with audio
    mp4_audio_path = fixtures_dir / "valid_with_audio.mp4"
    cmd_mp4_audio = [
        ffmpeg, "-nostdin", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:d=3",
        "-i", str(base_wav),
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        str(mp4_audio_path)
    ]
    subprocess.run(cmd_mp4_audio, stdin=subprocess.DEVNULL, capture_output=True, check=True)
    
    # 4. MP4 without audio (silent)
    mp4_silent_path = fixtures_dir / "silent_no_audio.mp4"
    cmd_mp4_silent = [
        ffmpeg, "-nostdin", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2",
        "-c:v", "libx264", "-an",
        str(mp4_silent_path)
    ]
    subprocess.run(cmd_mp4_silent, stdin=subprocess.DEVNULL, capture_output=True, check=True)
    
    # 5. Corrupted MP3
    corrupt_mp3_path = fixtures_dir / "corrupted_audio.mp3"
    corrupt_mp3_path.write_bytes(b"INVALID_HEADER_GARBAGE_BYTES_1234567890" * 50)
    
    # 6. Corrupted MP4
    corrupt_mp4_path = fixtures_dir / "corrupted_video.mp4"
    corrupt_mp4_path.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00CORRUPT_BYTES_XYZ" * 50)
    
    # 7. Unsupported format
    unsupported_path = fixtures_dir / "document.txt"
    unsupported_path.write_text("Plain text file.", encoding="utf-8")
    
    # 8. Empty audio file
    empty_path = fixtures_dir / "empty_sample.mp3"
    empty_path.write_bytes(b"")

    results_p1 = {}

    # Verification 1: WAV
    print("\n--- 1. Testing WAV Audio ---")
    feat_wav = _preprocess_audio(wav_path)
    print(f"Preprocessed shape: {feat_wav.shape}")
    assert feat_wav.shape == (1, 128, 94, 1), f"Unexpected shape {feat_wav.shape}"
    model_audio = _load_model("audio")
    raw_pred_wav = model_audio.predict(feat_wav, verbose=0).reshape(-1)
    prob_real_wav = float(raw_pred_wav[0])
    thresh_audio = get_threshold("audio")
    pred_label_wav = "Real" if prob_real_wav >= thresh_audio else "Fake"
    print(f"Inference: Probability(Real) = {prob_real_wav:.6f}, Threshold = {thresh_audio:.4f}, Label = {pred_label_wav}")
    res_wav = predict_file(wav_path, "audio")
    assert res_wav["prediction"] == pred_label_wav
    assert abs(res_wav["probability_real"] - prob_real_wav) < 1e-6
    results_p1["wav"] = {
        "shape": list(feat_wav.shape),
        "prob_real": prob_real_wav,
        "prediction": pred_label_wav,
        "confidence": res_wav["confidence"],
        "threshold": thresh_audio
    }
    print("  -> WAV verification: PASSED")

    # Verification 2: MP3
    print("\n--- 2. Testing MP3 Audio ---")
    # Check intermediate conversion
    temp_wav_mp3 = extract_audio_track_ffmpeg(mp3_path, target_sr=16000)
    assert temp_wav_mp3.is_file(), "Conversion failed to yield temporary WAV"
    import soundfile as sf
    data_mp3, sr_mp3 = sf.read(str(temp_wav_mp3))
    temp_wav_mp3.unlink(missing_ok=True)
    print(f"Converted WAV sample rate: {sr_mp3} Hz, Channels: {1 if data_mp3.ndim == 1 else data_mp3.shape[1]}")
    assert sr_mp3 == 16000, f"Expected 16000 Hz, got {sr_mp3}"
    assert data_mp3.ndim == 1 or data_mp3.shape[1] == 1, "Expected mono channel"
    feat_mp3 = _preprocess_audio(mp3_path)
    print(f"Preprocessed shape: {feat_mp3.shape}")
    assert feat_mp3.shape == (1, 128, 94, 1)
    raw_pred_mp3 = model_audio.predict(feat_mp3, verbose=0).reshape(-1)
    prob_real_mp3 = float(raw_pred_mp3[0])
    pred_label_mp3 = "Real" if prob_real_mp3 >= thresh_audio else "Fake"
    print(f"Inference: Probability(Real) = {prob_real_mp3:.6f}, Threshold = {thresh_audio:.4f}, Label = {pred_label_mp3}")
    res_mp3 = predict_file(mp3_path, "audio")
    assert res_mp3["prediction"] == pred_label_mp3
    results_p1["mp3"] = {
        "converted_sr": sr_mp3,
        "channels": 1,
        "shape": list(feat_mp3.shape),
        "prob_real": prob_real_mp3,
        "prediction": pred_label_mp3,
        "confidence": res_mp3["confidence"],
    }
    print("  -> MP3 verification: PASSED")

    # Verification 3: MP4 with Audio Track
    print("\n--- 3. Testing MP4 Containing Audio Track ---")
    temp_wav_mp4 = extract_audio_track_ffmpeg(mp4_audio_path, target_sr=16000)
    data_mp4, sr_mp4 = sf.read(str(temp_wav_mp4))
    temp_wav_mp4.unlink(missing_ok=True)
    print(f"Extracted audio track sample rate: {sr_mp4} Hz, Channels: {1 if data_mp4.ndim == 1 else data_mp4.shape[1]}")
    assert sr_mp4 == 16000
    assert data_mp4.ndim == 1 or data_mp4.shape[1] == 1
    feat_mp4 = _preprocess_audio(mp4_audio_path)
    print(f"Preprocessed shape: {feat_mp4.shape}")
    assert feat_mp4.shape == (1, 128, 94, 1)
    raw_pred_mp4 = model_audio.predict(feat_mp4, verbose=0).reshape(-1)
    prob_real_mp4 = float(raw_pred_mp4[0])
    pred_label_mp4 = "Real" if prob_real_mp4 >= thresh_audio else "Fake"
    print(f"Inference: Probability(Real) = {prob_real_mp4:.6f}, Threshold = {thresh_audio:.4f}, Label = {pred_label_mp4}")
    res_mp4 = predict_file(mp4_audio_path, "audio")
    assert res_mp4["prediction"] == pred_label_mp4
    results_p1["mp4_audio"] = {
        "extracted_sr": sr_mp4,
        "channels": 1,
        "shape": list(feat_mp4.shape),
        "prob_real": prob_real_mp4,
        "prediction": pred_label_mp4,
        "confidence": res_mp4["confidence"],
    }
    print("  -> MP4 with audio track verification: PASSED")

    # Verification 4: MP4 without Audio Track
    print("\n--- 4. Testing MP4 Without Audio Track (Silent) ---")
    try:
        predict_file(mp4_silent_path, "audio")
        assert False, "Silent MP4 should have been rejected"
    except Exception as exc:
        print(f"Caught expected rejection: {type(exc).__name__} -> {exc}")
        assert "audio track" in str(exc).lower()
        results_p1["mp4_silent"] = {"rejected": True, "error": str(exc)}
        print("  -> MP4 silent rejection verification: PASSED")

    # Verification 5: Corrupted MP3
    print("\n--- 5. Testing Corrupted MP3 ---")
    try:
        predict_file(corrupt_mp3_path, "audio")
        assert False, "Corrupted MP3 should have been rejected"
    except Exception as exc:
        print(f"Caught expected rejection: {type(exc).__name__} -> {exc}")
        assert "could not be decoded" in str(exc).lower()
        results_p1["corrupt_mp3"] = {"rejected": True, "error": str(exc)}
        print("  -> Corrupted MP3 rejection verification: PASSED")

    # Verification 6: Corrupted MP4
    print("\n--- 6. Testing Corrupted MP4 ---")
    try:
        predict_file(corrupt_mp4_path, "audio")
        assert False, "Corrupted MP4 should have been rejected"
    except Exception as exc:
        print(f"Caught expected rejection: {type(exc).__name__} -> {exc}")
        assert "could not be decoded" in str(exc).lower() or "audio track" in str(exc).lower()
        results_p1["corrupt_mp4"] = {"rejected": True, "error": str(exc)}
        print("  -> Corrupted MP4 rejection verification: PASSED")

    # Verification 7: Unsupported format
    print("\n--- 7. Testing Unsupported Format (.txt) ---")
    try:
        predict_file(unsupported_path, "audio")
        assert False, "Unsupported format should have been rejected"
    except Exception as exc:
        print(f"Caught expected rejection: {type(exc).__name__} -> {exc}")
        assert "not supported" in str(exc).lower()
        results_p1["unsupported"] = {"rejected": True, "error": str(exc)}
        print("  -> Unsupported format rejection verification: PASSED")

    # Verification 8: Empty file
    print("\n--- 8. Testing Empty Audio File (0-byte) ---")
    try:
        predict_file(empty_path, "audio")
        assert False, "Empty audio file should have been rejected"
    except Exception as exc:
        print(f"Caught expected rejection: {type(exc).__name__} -> {exc}")
        assert "empty" in str(exc).lower() or "could not be decoded" in str(exc).lower()
        results_p1["empty"] = {"rejected": True, "error": str(exc)}
        print("  -> Empty audio file rejection verification: PASSED")

    report_data["phase1_and_2"] = results_p1


def run_phase2_web_workflow():
    print("\n" + "=" * 80)
    print("PHASE 2: WEB APPLICATION WORKFLOW & DATABASE HISTORY VERIFICATION")
    print("=" * 80)
    
    app.config["TESTING"] = True
    client = app.test_client()
    
    # Ensure test user in MySQL
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, username FROM users WHERE email='readonly_tester@example.com' LIMIT 1")
    user = cur.fetchone()
    if not user:
        cur.execute(
            "INSERT INTO users (fullname, username, email, mobile, password) VALUES (%s, %s, %s, %s, %s)",
            ("ReadOnly Tester", "readonly_tester", "readonly_tester@example.com", "9123456780", "Password123!")
        )
        conn.commit()
        user_id = cur.lastrowid
        user_name = "readonly_tester"
    else:
        user_id = user["id"]
        user_name = user["username"]
    cur.close()
    conn.close()
    
    with client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["username"] = user_name

    web_results = {}
    
    # Upload WAV
    with open(fixtures_dir / "valid_sample.wav", "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "ro_verify.wav", "audio/wav")}, follow_redirects=False)
        assert resp.status_code == 302 and "/result/" in resp.headers.get("Location")
        loc_wav = resp.headers.get("Location")
        pid_wav = int(loc_wav.split("/result/")[1])
        res_page = client.get(loc_wav)
        assert res_page.status_code == 200
        html = res_page.data.decode("utf-8")
        assert "Audio" in html
        assert "<audio" in html.lower()
        audit = get_prediction_audit(pid_wav)
        assert audit is not None and audit["file_type"] == "Audio"
        web_results["wav_upload"] = {"id": pid_wav, "status": 200, "confidence": audit["confidence"]}
        print(f"  -> WAV Web Upload & Result: HTTP 200, ID {pid_wav}, Confidence {audit['confidence']}%")

    # Upload MP3
    with open(fixtures_dir / "valid_sample.mp3", "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "ro_verify.mp3", "audio/mpeg")}, follow_redirects=False)
        assert resp.status_code == 302 and "/result/" in resp.headers.get("Location")
        loc_mp3 = resp.headers.get("Location")
        pid_mp3 = int(loc_mp3.split("/result/")[1])
        res_page = client.get(loc_mp3)
        assert res_page.status_code == 200
        html = res_page.data.decode("utf-8")
        assert "Audio" in html
        assert "<audio" in html.lower()
        audit = get_prediction_audit(pid_mp3)
        assert audit is not None and audit["file_type"] == "Audio"
        web_results["mp3_upload"] = {"id": pid_mp3, "status": 200, "confidence": audit["confidence"]}
        print(f"  -> MP3 Web Upload & Result: HTTP 200, ID {pid_mp3}, Confidence {audit['confidence']}%")

    # Upload MP4 with Audio
    with open(fixtures_dir / "valid_with_audio.mp4", "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "ro_verify.mp4", "video/mp4")}, follow_redirects=False)
        assert resp.status_code == 302 and "/result/" in resp.headers.get("Location")
        loc_mp4 = resp.headers.get("Location")
        pid_mp4 = int(loc_mp4.split("/result/")[1])
        res_page = client.get(loc_mp4)
        assert res_page.status_code == 200
        html = res_page.data.decode("utf-8")
        assert "Audio" in html
        assert "<audio" in html.lower()
        audit = get_prediction_audit(pid_mp4)
        assert audit is not None and audit["file_type"] == "Audio"
        web_results["mp4_upload"] = {"id": pid_mp4, "status": 200, "confidence": audit["confidence"]}
        print(f"  -> MP4 Web Upload & Result: HTTP 200, ID {pid_mp4}, Confidence {audit['confidence']}%")

    # Verify History Table
    hist_resp = client.get("/history")
    assert hist_resp.status_code == 200
    hist_html = hist_resp.data.decode("utf-8")
    assert "ro_verify.wav" in hist_html
    assert "ro_verify.mp3" in hist_html
    assert "ro_verify.mp4" in hist_html
    print("  -> History page properly rendered all 3 audio upload entries.")

    report_data["web_audio"] = web_results


def run_phase3_regression():
    print("\n" + "=" * 80)
    print("PHASE 3: REGRESSION TESTING (IMAGE, VIDEO, AUTH, DASHBOARD)")
    print("=" * 80)
    
    app.config["TESTING"] = True
    client = app.test_client()
    
    reg_results = {}
    
    # 1. Image Detection Regression
    print("\n--- 1. Image Detection Regression ---")
    hr_img = Path("D:/deepfake_college_project/ai-generated-8622236_1280.jpg")
    lr_img = Path("D:/deepfake_college_project/OIP.jpg")
    
    route_hr = route_image(hr_img)
    print(f"High-Res Image Routing: Route = {route_hr['route']}, Version = {route_hr['model_version']}, Threshold = {route_hr['threshold']}")
    assert route_hr["route"] == "v3"
    assert route_hr["threshold"] == 0.3800
    pred_hr = predict_file(hr_img, "image")
    print(f"High-Res Prediction: {pred_hr['prediction']}, Conf: {pred_hr['confidence']:.4f}, Model: {pred_hr['model_used']}")
    assert pred_hr["model_version"] == "V3"
    assert pred_hr["prediction"] == "Fake"

    route_lr = route_image(lr_img)
    print(f"Low-Res Image Routing: Route = {route_lr['route']}, Version = {route_lr['model_version']}, Threshold = {route_lr['threshold']}")
    assert route_lr["route"] == "v2"
    assert route_lr["threshold"] == 0.5200
    pred_lr = predict_file(lr_img, "image")
    print(f"Low-Res Prediction: {pred_lr['prediction']}, Conf: {pred_lr['confidence']:.4f}, Model: {pred_lr['model_used']}")
    assert pred_lr["model_version"] == "V2"
    assert pred_lr["prediction"] == "Fake"
    
    reg_results["image"] = {
        "hr_route": route_hr["route"],
        "hr_threshold": route_hr["threshold"],
        "hr_prediction": pred_hr["prediction"],
        "lr_route": route_lr["route"],
        "lr_threshold": route_lr["threshold"],
        "lr_prediction": pred_lr["prediction"],
    }
    print("  -> Image regression: PASSED")

    # 2. Video Detection Regression
    print("\n--- 2. Video Detection Regression ---")
    vid_file = base_dir / "dataset" / "video" / "test" / "fake" / "00001_id01683_wavtolip.mp4"
    assert vid_file.is_file(), f"Video test file missing: {vid_file}"
    vid_thresh = get_threshold("video")
    print(f"Video Threshold: {vid_thresh}")
    assert vid_thresh == 0.5500
    pred_vid = predict_file(vid_file, "video")
    print(f"Video Prediction: {pred_vid['prediction']}, Conf: {pred_vid['confidence']:.4f}, Frames: {pred_vid.get('frames_analyzed')}")
    assert pred_vid["modality"] == "video"
    assert pred_vid["prediction"] in {"Real", "Fake"}
    reg_results["video"] = {
        "threshold": vid_thresh,
        "prediction": pred_vid["prediction"],
        "confidence": pred_vid["confidence"],
        "frames": pred_vid.get("frames_analyzed")
    }
    print("  -> Video regression: PASSED")

    # 3. Web App Authentication & Dashboard
    print("\n--- 3. Web Application, Auth & Dashboard Regression ---")
    # Register new random user
    reg_user = f"reg_user_{uuid.uuid4().hex[:6]}"
    reg_email = f"{reg_user}@example.com"
    resp_reg = client.post("/register", data={
        "fullname": "Regression Tester",
        "username": reg_user,
        "email": reg_email,
        "mobile": "9876500000",
        "password": "Password123!"
    }, follow_redirects=False)
    assert resp_reg.status_code == 302 and "/login" in resp_reg.headers.get("Location")
    print(f"Registration for {reg_email}: SUCCESS (redirects to /login)")

    # Login
    resp_login = client.post("/login", data={
        "email": reg_email,
        "password": "Password123!"
    }, follow_redirects=False)
    assert resp_login.status_code == 302 and "/dashboard" in resp_login.headers.get("Location")
    print(f"Login for {reg_email}: SUCCESS (redirects to /dashboard)")

    # Dashboard access
    resp_dash = client.get("/dashboard")
    assert resp_dash.status_code == 200
    dash_html = resp_dash.data.decode("utf-8")
    assert "Dashboard" in dash_html
    print("Dashboard access: SUCCESS (HTTP 200)")

    # CSV Export
    resp_csv = client.get("/export_csv")
    assert resp_csv.status_code == 200
    assert "text/csv" in resp_csv.content_type
    print("CSV Export: SUCCESS (HTTP 200, text/csv)")

    reg_results["web_auth"] = {
        "registration": "SUCCESS",
        "login": "SUCCESS",
        "dashboard": "SUCCESS",
        "csv_export": "SUCCESS"
    }
    print("  -> Web Application regression: PASSED")
    report_data["regression"] = reg_results


def run_phase4_hashes():
    print("\n" + "=" * 80)
    print("PHASE 4: MODEL AND CALIBRATION PRESERVATION (SHA-256 HASHES)")
    print("=" * 80)
    
    files = {
        "audio_model": "models/saved_models/audio/best_model.h5",
        "audio_calibration": "models/calibration.json",
        "image_v2_model": "models/saved_models/image/best_model_v2.h5",
        "image_v2_calibration": "models/calibration.json",
        "image_v3_model": "models/saved_models/image/best_model_v3.keras",
        "image_v3_calibration": "models/calibration_v3.json",
        "video_model": "models/saved_models/video/best_model.h5",
        "video_calibration": "models/calibration.json",
    }
    
    hashes = {}
    for key, rel_path in files.items():
        p = base_dir / rel_path
        assert p.is_file(), f"Required file missing: {p}"
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        size = p.stat().st_size
        hashes[key] = {"path": rel_path, "size": size, "sha256": sha}
        print(f"{key:<22} | {rel_path:<45} | Size: {size:>10,} B | SHA256: {sha}")
        
    report_data["hashes"] = hashes


if __name__ == "__main__":
    run_phase1_and_phase2()
    run_phase2_web_workflow()
    run_phase3_regression()
    run_phase4_hashes()
    
    out_json = base_dir / "runtime" / "temp" / "verification_results.json"
    out_json.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    print("\nVerification completed successfully. JSON saved to:", out_json)
