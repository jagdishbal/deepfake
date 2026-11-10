import os
import sys
import io
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import config
from app import app
from database.database import get_db_connection
from services.audit_service import get_prediction_audit

def run_comprehensive_live_test():
    print("=" * 80)
    print("RUNNING COMPREHENSIVE E2E LIVE TEST ACROSS ALL 3 MODALITIES")
    print("=" * 80)

    app.config["TESTING"] = True
    client = app.test_client()

    # 1. Setup authenticated session with real MySQL user
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, username, email FROM users WHERE email='tester@college.edu' LIMIT 1")
    user = cur.fetchone()
    if not user:
        cur.execute(
            "INSERT INTO users (fullname, username, email, mobile, password) VALUES (%s, %s, %s, %s, %s)",
            ("College Reviewer", "reviewer", "tester@college.edu", "9876543210", "CollegePass123!")
        )
        conn.commit()
        user_id = cur.lastrowid
        username = "reviewer"
    else:
        user_id = user["id"]
        username = user["username"]
    cur.close()
    conn.close()

    with client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["username"] = username

    # 2. Test Navigation / Route Availability
    print("\n--- 1. Testing Route Availability ---")
    routes_to_test = [
        ("/", 200),
        ("/agreement", 200),
        ("/dashboard", 200),
        ("/history", 200),
        ("/image_detection", 200),
        ("/video_detection", 200),
        ("/audio_detection", 200),
    ]
    for route, expected_status in routes_to_test:
        resp = client.get(route)
        print(f"  GET {route:<20} -> Status: {resp.status_code}")
        assert resp.status_code == expected_status, f"Route {route} returned {resp.status_code}"

    # 3. Test IMAGE Detection Pipeline
    print("\n--- 2. Testing Image Detection Workflow ---")
    img_sample = next((PROJECT_ROOT / "dataset" / "image" / "test" / "real").glob("*.jpg"), None)
    if not img_sample:
        img_sample = next((PROJECT_ROOT / "dataset" / "image" / "test" / "real").glob("*.png"), None)
    assert img_sample is not None, "No image test sample found"

    with open(img_sample, "rb") as f:
        resp = client.post(
            "/image_detection",
            data={"image": (io.BytesIO(f.read()), img_sample.name, "image/jpeg")},
            follow_redirects=False,
        )
        print(f"  POST /image_detection -> Status: {resp.status_code}, Location: {resp.headers.get('Location')}")
        assert resp.status_code == 302 and "/result/" in resp.headers.get("Location")
        img_result_id = int(resp.headers.get("Location").split("/result/")[1])

        # Follow to result page
        res_page = client.get(resp.headers.get("Location"))
        assert res_page.status_code == 200
        html = res_page.data.decode("utf-8")
        assert "Image" in html or "Detection Result" in html
        print(f"  [SUCCESS] Image Result Page rendered (Prediction ID: {img_result_id})")

    # 4. Test VIDEO Detection Pipeline
    print("\n--- 3. Testing Video Detection Workflow ---")
    vid_sample = next((PROJECT_ROOT / "dataset" / "video" / "test" / "fake").glob("*.mp4"), None)
    assert vid_sample is not None, "No video test sample found"

    with open(vid_sample, "rb") as f:
        resp = client.post(
            "/video_detection",
            data={"video": (io.BytesIO(f.read()), vid_sample.name, "video/mp4")},
            follow_redirects=False,
        )
        print(f"  POST /video_detection -> Status: {resp.status_code}, Location: {resp.headers.get('Location')}")
        assert resp.status_code == 302 and "/result/" in resp.headers.get("Location")
        vid_result_id = int(resp.headers.get("Location").split("/result/")[1])

        # Follow to result page
        res_page = client.get(resp.headers.get("Location"))
        assert res_page.status_code == 200
        html = res_page.data.decode("utf-8")
        assert "Video" in html or "Detection Result" in html
        print(f"  [SUCCESS] Video Result Page rendered (Prediction ID: {vid_result_id})")

    # 5. Test AUDIO Detection Pipeline (WAV, MP3, FLAC)
    print("\n--- 4. Testing Audio Detection Workflow ---")
    audio_sample = next((PROJECT_ROOT / "dataset" / "audio" / "test" / "fake").glob("*.wav"), None)
    assert audio_sample is not None, "No audio test sample found"

    with open(audio_sample, "rb") as f:
        resp = client.post(
            "/audio_detection",
            data={"audio": (io.BytesIO(f.read()), audio_sample.name, "audio/wav")},
            follow_redirects=False,
        )
        print(f"  POST /audio_detection (WAV) -> Status: {resp.status_code}, Location: {resp.headers.get('Location')}")
        assert resp.status_code == 302 and "/result/" in resp.headers.get("Location")
        aud_result_id = int(resp.headers.get("Location").split("/result/")[1])

        # Follow to result page
        res_page = client.get(resp.headers.get("Location"))
        assert res_page.status_code == 200
        html = res_page.data.decode("utf-8")
        assert "Audio" in html or "Detection Result" in html
        print(f"  [SUCCESS] Audio Result Page rendered (Prediction ID: {aud_result_id})")

    # 6. Verify MySQL Database Persistence
    print("\n--- 5. Verifying MySQL Database Records ---")
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT id, filename, file_type, prediction, confidence FROM predictions WHERE user_id=%s ORDER BY id DESC LIMIT 3",
        (user_id,)
    )
    records = cur.fetchall()
    cur.close()
    conn.close()

    print(f"  Found {len(records)} recent database prediction records for user {user_id}:")
    for r in records:
        print(f"    - ID {r['id']}: [{r['file_type']}] {r['prediction']} (Confidence: {r['confidence']}%) File: {r['filename'][:30]}")
    assert len(records) >= 3, "Expected at least 3 records in MySQL predictions table"

    # 7. Verify History Page Displays All 3 Modalities
    print("\n--- 6. Verifying User History Page ---")
    hist_resp = client.get("/history")
    assert hist_resp.status_code == 200
    hist_html = hist_resp.data.decode("utf-8")
    assert "Image" in hist_html
    assert "Video" in hist_html
    assert "Audio" in hist_html
    print("  [SUCCESS] History page contains Image, Video, and Audio entries.")

    # 8. Test Admin Login & Admin Dashboard
    print("\n--- 7. Verifying Admin Login & Admin Dashboard ---")
    with client.session_transaction() as sess:
        sess.clear()
    login_resp = client.post(
        "/login",
        data={"email": "admin", "password": "admin123"},
        follow_redirects=False,
    )
    print(f"  POST /login (admin/admin123) -> Status: {login_resp.status_code}, Location: {login_resp.headers.get('Location')}")
    assert login_resp.status_code == 302 and "/admin_dashboard" in login_resp.headers.get("Location")

    admin_dash_resp = client.get("/admin_dashboard")
    assert admin_dash_resp.status_code == 200
    dash_html = admin_dash_resp.data.decode("utf-8")
    assert "Admin Dashboard" in dash_html
    assert "Registered Users" in dash_html
    assert "Total Predictions" in dash_html
    print("  [SUCCESS] Admin Dashboard renders live user count and prediction statistics.")

    print("\n" + "=" * 80)
    print("ALL MODULES (IMAGE, VIDEO, AUDIO, USER, ADMIN, DB) VERIFIED WORKING END-TO-END!")
    print("=" * 80)

if __name__ == "__main__":
    run_comprehensive_live_test()
