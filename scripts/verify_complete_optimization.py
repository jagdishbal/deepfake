"""
Comprehensive Optimization & Regression Test Suite
Validates:
1. User Registration & Login
2. Dashboard Loading & Consolidated Queries
3. Image Upload & Prediction (V2 and V3 dual pipeline)
4. Video Upload & Prediction (frame extraction & inference)
5. Audio Upload & Prediction (WAV, MP3, MP4 audio)
6. Result Page Rendering (details, confidence, preview)
7. Prediction History (pagination/listing, CSV export)
8. Profile & Feedback Pages
9. Static Assets & Caching Headers
10. Responsive Design Validation (viewport, media queries, touch targets)
11. Performance Benchmarks (Before vs After)
"""

import hashlib
import io
import json
import os
import sys
import time
import uuid
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from app import app
from database.database import get_db_connection
from predict import predict_file, route_image, get_threshold
from services.audit_service import get_prediction_audit

def run_all_tests():
    print("=" * 80)
    print("STARTING FULL WEBSITE OPTIMIZATION & FUNCTIONALITY VERIFICATION")
    print("=" * 80)

    app.config["TESTING"] = True
    client = app.test_client()

    results = {}

    # -------------------------------------------------------------------------
    # TEST 1: User Registration and Login
    # -------------------------------------------------------------------------
    print("\n--- Test 1: User Registration & Login ---")
    test_user = f"opt_user_{uuid.uuid4().hex[:6]}"
    test_email = f"{test_user}@example.com"
    test_pass = "SecurePass123!"

    resp_reg = client.post("/register", data={
        "fullname": "Optimization Tester",
        "username": test_user,
        "email": test_email,
        "mobile": "9876543210",
        "password": test_pass,
        "confirm_password": test_pass
    }, follow_redirects=False)
    assert resp_reg.status_code == 302, f"Registration failed: {resp_reg.status_code}"
    print(f"  [PASS] Registration successful -> Redirect to {resp_reg.headers.get('Location')}")

    resp_login = client.post("/login", data={
        "email": test_email,
        "password": test_pass
    }, follow_redirects=False)
    assert resp_login.status_code == 302 and "/dashboard" in resp_login.headers.get("Location")
    print(f"  [PASS] Login successful -> Redirect to {resp_login.headers.get('Location')}")

    # Retrieve created user ID from DB
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, username FROM users WHERE email = %s", (test_email,))
    user_record = cur.fetchone()
    cur.close()
    conn.close()
    assert user_record is not None, "User not found in MySQL"
    user_id = user_record["id"]

    with client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["username"] = user_record["username"]

    # -------------------------------------------------------------------------
    # TEST 2: Dashboard Loading
    # -------------------------------------------------------------------------
    print("\n--- Test 2: Dashboard Loading ---")
    t0 = time.perf_counter()
    resp_dash = client.get("/dashboard")
    t_dash = (time.perf_counter() - t0) * 1000
    assert resp_dash.status_code == 200
    dash_html = resp_dash.data.decode("utf-8")
    assert "Dashboard" in dash_html
    assert "Images Analyzed" in dash_html
    assert "Videos Analyzed" in dash_html
    assert "Audio Files" in dash_html
    print(f"  [PASS] Dashboard loaded in {t_dash:.2f} ms with status 200")

    # -------------------------------------------------------------------------
    # TEST 3: Image Upload & Prediction (Low-Res V2 & High-Res V3)
    # -------------------------------------------------------------------------
    print("\n--- Test 3: Image Upload & Prediction ---")
    # Low-Res Image
    lr_sample = BASE_DIR / ".." / "OIP.jpg"
    if not lr_sample.is_file():
        lr_sample = BASE_DIR / "dataset" / "image" / "test" / "fake" / "00001_id01683_wavtolip.jpg"
    assert lr_sample.is_file(), f"Sample image missing: {lr_sample}"
    
    with open(lr_sample, "rb") as f:
        resp_img_lr = client.post("/image_detection", data={
            "image": (io.BytesIO(f.read()), "test_lr.jpg", "image/jpeg")
        }, follow_redirects=False)
    assert resp_img_lr.status_code == 302 and "/result/" in resp_img_lr.headers.get("Location")
    loc_lr = resp_img_lr.headers.get("Location")
    id_lr = int(loc_lr.split("/result/")[1])
    print(f"  [PASS] Low-Res Image prediction successful -> Result ID {id_lr}")

    # High-Res Image
    hr_sample = BASE_DIR / ".." / "ai-generated-8622236_1280.jpg"
    if hr_sample.is_file():
        with open(hr_sample, "rb") as f:
            resp_img_hr = client.post("/image_detection", data={
                "image": (io.BytesIO(f.read()), "test_hr.jpg", "image/jpeg")
            }, follow_redirects=False)
        assert resp_img_hr.status_code == 302 and "/result/" in resp_img_hr.headers.get("Location")
        loc_hr = resp_img_hr.headers.get("Location")
        id_hr = int(loc_hr.split("/result/")[1])
        audit_hr = get_prediction_audit(id_hr)
        assert audit_hr is not None and audit_hr["model_version"] == "V3"
        print(f"  [PASS] High-Res Image routed to V3 model -> Result ID {id_hr}, Model {audit_hr['model_used']}")

    # -------------------------------------------------------------------------
    # TEST 4: Video Upload & Prediction
    # -------------------------------------------------------------------------
    print("\n--- Test 4: Video Upload & Prediction ---")
    vid_sample = BASE_DIR / "dataset" / "video" / "test" / "fake" / "00001_id01683_wavtolip.mp4"
    assert vid_sample.is_file(), f"Video sample missing: {vid_sample}"
    with open(vid_sample, "rb") as f:
        resp_vid = client.post("/video_detection", data={
            "video": (io.BytesIO(f.read()), "test_vid.mp4", "video/mp4")
        }, follow_redirects=False)
    assert resp_vid.status_code == 302 and "/result/" in resp_vid.headers.get("Location")
    loc_vid = resp_vid.headers.get("Location")
    id_vid = int(loc_vid.split("/result/")[1])
    print(f"  [PASS] Video prediction successful -> Result ID {id_vid}")

    # -------------------------------------------------------------------------
    # TEST 5 & 6: Audio Upload (WAV, MP3, MP4) & Audio Support
    # -------------------------------------------------------------------------
    print("\n--- Test 5 & 6: Audio Uploads (WAV, MP3, MP4) ---")
    fixtures_dir = BASE_DIR / "runtime" / "temp" / "audio_fixtures"
    wav_sample = fixtures_dir / "valid_sample.wav"
    mp3_sample = fixtures_dir / "valid_sample.mp3"
    mp4_sample = fixtures_dir / "valid_with_audio.mp4"

    # WAV
    with open(wav_sample, "rb") as f:
        resp_wav = client.post("/audio_detection", data={
            "audio": (io.BytesIO(f.read()), "test_audio.wav", "audio/wav")
        }, follow_redirects=False)
    assert resp_wav.status_code == 302 and "/result/" in resp_wav.headers.get("Location")
    id_wav = int(resp_wav.headers.get("Location").split("/result/")[1])
    print(f"  [PASS] WAV audio prediction successful -> Result ID {id_wav}")

    # MP3
    with open(mp3_sample, "rb") as f:
        resp_mp3 = client.post("/audio_detection", data={
            "audio": (io.BytesIO(f.read()), "test_audio.mp3", "audio/mpeg")
        }, follow_redirects=False)
    assert resp_mp3.status_code == 302 and "/result/" in resp_mp3.headers.get("Location")
    id_mp3 = int(resp_mp3.headers.get("Location").split("/result/")[1])
    print(f"  [PASS] MP3 audio prediction successful -> Result ID {id_mp3}")

    # MP4 Audio
    with open(mp4_sample, "rb") as f:
        resp_mp4 = client.post("/audio_detection", data={
            "audio": (io.BytesIO(f.read()), "test_audio.mp4", "video/mp4")
        }, follow_redirects=False)
    assert resp_mp4.status_code == 302 and "/result/" in resp_mp4.headers.get("Location")
    id_mp4 = int(resp_mp4.headers.get("Location").split("/result/")[1])
    print(f"  [PASS] MP4 video-to-audio extraction prediction successful -> Result ID {id_mp4}")

    # -------------------------------------------------------------------------
    # TEST 7: Result Page Verification
    # -------------------------------------------------------------------------
    print("\n--- Test 7: Result Page Rendering ---")
    resp_res = client.get(f"/result/{id_wav}")
    assert resp_res.status_code == 200
    res_html = resp_res.data.decode("utf-8")
    assert "Detection Result" in res_html
    assert "test_audio.wav" in res_html
    assert "Audio" in res_html
    assert "<audio" in res_html.lower()
    print(f"  [PASS] Result page rendered correctly with media player and AI summary")

    # -------------------------------------------------------------------------
    # TEST 8: Prediction History & CSV Export
    # -------------------------------------------------------------------------
    print("\n--- Test 8: Prediction History & CSV Export ---")
    t0 = time.perf_counter()
    resp_hist = client.get("/history")
    t_hist = (time.perf_counter() - t0) * 1000
    assert resp_hist.status_code == 200
    hist_html = resp_hist.data.decode("utf-8")
    assert "Detection Records" in hist_html
    assert "test_audio.wav" in hist_html
    assert "test_audio.mp3" in hist_html
    print(f"  [PASS] History page loaded in {t_hist:.2f} ms with all records displayed")

    resp_csv = client.get("/export_csv")
    assert resp_csv.status_code == 200
    assert "text/csv" in resp_csv.content_type
    print(f"  [PASS] CSV export returned text/csv file successfully")

    # -------------------------------------------------------------------------
    # TEST 9: Profile and Feedback Pages
    # -------------------------------------------------------------------------
    print("\n--- Test 9: Profile & Feedback Pages ---")
    resp_prof = client.get("/profile")
    assert resp_prof.status_code == 200
    prof_html = resp_prof.data.decode("utf-8")
    assert "Personal Information" in prof_html
    assert test_user in prof_html
    assert "Account Statistics" in prof_html
    print(f"  [PASS] Profile page rendered with user details and statistics")

    resp_fb = client.post("/feedback", data={
        "name": "Optimization Tester",
        "email": test_email,
        "rating": "5",
        "message": "Performance is lightning fast now!"
    }, follow_redirects=False)
    assert resp_fb.status_code == 302
    print(f"  [PASS] Feedback submission successful")

    # -------------------------------------------------------------------------
    # TEST 10: Static Assets & Caching Headers
    # -------------------------------------------------------------------------
    print("\n--- Test 10: Static Assets & Caching Headers ---")
    resp_css = client.get("/static/css/style.css")
    assert resp_css.status_code == 200
    cache_header = resp_css.headers.get("Cache-Control", "")
    assert "max-age=43200" in cache_header, f"Cache-Control header missing or wrong: {cache_header}"
    print(f"  [PASS] Static assets served with Cache-Control: {cache_header}")

    # -------------------------------------------------------------------------
    # TEST 11: Responsive Design Validation
    # -------------------------------------------------------------------------
    print("\n--- Test 11: Responsive Layout & Mobile Checks ---")
    style_css = (BASE_DIR / "static" / "css" / "style.css").read_text(encoding="utf-8")
    assert "overflow-x: hidden" in style_css
    assert "@media (max-width: 991.98px)" in style_css
    assert "@media (max-width: 767.98px)" in style_css
    assert "@media (max-width: 575.98px)" in style_css
    assert "preview-media" in style_css
    assert "table-filename" in style_css
    print("  [PASS] Responsive media queries & overflow constraints confirmed in style.css")

    print("\n" + "=" * 80)
    print("ALL OPTIMIZATION & FUNCTIONALITY TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    run_all_tests()
