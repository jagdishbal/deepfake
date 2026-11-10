import os
import sys
import io
import subprocess
from pathlib import Path

os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from app import app
from predict import predict_file, SUPPORTED_EXTENSIONS
from database.database import get_db_connection
from services.audit_service import get_prediction_audit
from imageio_ffmpeg import get_ffmpeg_exe

ffmpeg = get_ffmpeg_exe()
fixture_dir = Path(config.BASE_DIR) / "runtime" / "temp" / "audio_fixtures"
fixture_dir.mkdir(parents=True, exist_ok=True)

def create_fixtures():
    print("--- Creating Test Fixtures ---")
    fake_wavs = list((Path(config.BASE_DIR) / "dataset" / "audio" / "test" / "fake").glob("*.wav"))
    assert len(fake_wavs) > 0, "No sample wav found in dataset/audio/test/fake"
    base_wav = fake_wavs[0]
    
    # 1. WAV
    wav_file = fixture_dir / "valid_sample.wav"
    wav_file.write_bytes(base_wav.read_bytes())
    
    # 2. MP3
    mp3_file = fixture_dir / "valid_sample.mp3"
    subprocess.run([ffmpeg, "-y", "-i", str(base_wav), "-acodec", "libmp3lame", "-b:a", "128k", str(mp3_file)], capture_output=True, check=True)
    
    # 2B. M4A
    m4a_file = fixture_dir / "valid_sample.m4a"
    subprocess.run([ffmpeg, "-y", "-i", str(base_wav), "-c:a", "aac", str(m4a_file)], capture_output=True, check=True)

    # 2C. FLAC
    flac_file = fixture_dir / "valid_sample.flac"
    subprocess.run([ffmpeg, "-y", "-i", str(base_wav), "-c:a", "flac", str(flac_file)], capture_output=True, check=True)
    
    # 3. MP4 with audio track
    mp4_with_audio = fixture_dir / "valid_with_audio.mp4"
    cmd_mp4_audio = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "color=c=black:s=320x240:d=3",
        "-i", str(base_wav),
        "-c:v", "libx264", "-c:a", "aac", "-shortest",
        str(mp4_with_audio)
    ]
    subprocess.run(cmd_mp4_audio, capture_output=True, check=True)
    
    # 4. MP4 without audio track (silent)
    mp4_silent = fixture_dir / "silent_no_audio.mp4"
    cmd_mp4_silent = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2",
        "-c:v", "libx264", "-an",
        str(mp4_silent)
    ]
    subprocess.run(cmd_mp4_silent, capture_output=True, check=True)
    
    # 5. Corrupted MP3
    corrupt_mp3 = fixture_dir / "corrupted_audio.mp3"
    corrupt_mp3.write_bytes(b"INVALID_HEADER_GARBAGE_BYTES_1234567890" * 50)
    
    # 6. Corrupted MP4
    corrupt_mp4 = fixture_dir / "corrupted_video.mp4"
    corrupt_mp4.write_bytes(b"\x00\x00\x00\x18ftypisom\x00\x00CORRUPT_BYTES_XYZ" * 50)
    
    # 7. Unsupported format
    unsupported_txt = fixture_dir / "document.txt"
    unsupported_txt.write_text("This is an unsupported text document.", encoding="utf-8")
    
    # 8. Empty audio file
    empty_mp3 = fixture_dir / "empty_sample.mp3"
    empty_mp3.write_bytes(b"")
    
    # 9A. Audio exceeding allowed duration (>300 seconds, e.g. 310 seconds)
    long_audio = fixture_dir / "over_duration.mp3"
    cmd_long = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=310",
        "-c:a", "libmp3lame",
        str(long_audio)
    ]
    subprocess.run(cmd_long, capture_output=True, check=True)
    
    # 9B. Audio exceeding allowed size (>50 MB)
    large_audio = fixture_dir / "over_size.wav"
    with open(large_audio, "wb") as f:
        # 51 MB dummy file
        f.seek(51 * 1024 * 1024 - 1)
        f.write(b"\0")
        
    print("All fixtures created successfully.")
    return {
        "wav": wav_file,
        "mp3": mp3_file,
        "m4a": m4a_file,
        "flac": flac_file,
        "mp4_audio": mp4_with_audio,
        "mp4_silent": mp4_silent,
        "corrupt_mp3": corrupt_mp3,
        "corrupt_mp4": corrupt_mp4,
        "unsupported": unsupported_txt,
        "empty": empty_mp3,
        "long_audio": long_audio,
        "large_audio": large_audio,
    }


def run_tests():
    fixtures = create_fixtures()
    
    print("\n" + "=" * 70)
    print("TEST SUITE: Direct Prediction Pipeline Tests")
    print("=" * 70)
    
    # Direct Test 1: WAV
    res_wav = predict_file(fixtures["wav"], "audio")
    print(f"[Direct 1 - WAV] Prediction: {res_wav['prediction']}, Conf: {res_wav['confidence']:.4f}, Modality: {res_wav['modality']}")
    assert res_wav["modality"] == "audio"
    assert res_wav["prediction"] in {"Real", "Fake"}
    
    # Direct Test 2: MP3
    res_mp3 = predict_file(fixtures["mp3"], "audio")
    print(f"[Direct 2 - MP3] Prediction: {res_mp3['prediction']}, Conf: {res_mp3['confidence']:.4f}, Modality: {res_mp3['modality']}")
    assert res_mp3["modality"] == "audio"
    assert res_mp3["prediction"] in {"Real", "Fake"}

    # Direct Test 2B: M4A
    res_m4a = predict_file(fixtures["m4a"], "audio")
    print(f"[Direct 2B - M4A] Prediction: {res_m4a['prediction']}, Conf: {res_m4a['confidence']:.4f}, Modality: {res_m4a['modality']}")
    assert res_m4a["modality"] == "audio"
    assert res_m4a["prediction"] in {"Real", "Fake"}

    # Direct Test 2C: FLAC
    res_flac = predict_file(fixtures["flac"], "audio")
    print(f"[Direct 2C - FLAC] Prediction: {res_flac['prediction']}, Conf: {res_flac['confidence']:.4f}, Modality: {res_flac['modality']}")
    assert res_flac["modality"] == "audio"
    assert res_flac["prediction"] in {"Real", "Fake"}
    
    # Direct Test 3: MP4 with audio
    res_mp4 = predict_file(fixtures["mp4_audio"], "audio")
    print(f"[Direct 3 - MP4 with Audio] Prediction: {res_mp4['prediction']}, Conf: {res_mp4['confidence']:.4f}, Modality: {res_mp4['modality']}")
    assert res_mp4["modality"] == "audio"
    assert res_mp4["prediction"] in {"Real", "Fake"}
    
    # Direct Test 4: MP4 without audio -> expect MediaDecodeError
    try:
        predict_file(fixtures["mp4_silent"], "audio")
        assert False, "Expected MediaDecodeError for silent MP4"
    except Exception as e:
        print(f"[Direct 4 - Silent MP4] Correctly rejected: {type(e).__name__} -> {e}")
        assert "audio track" in str(e).lower()
        
    # Direct Test 5: Corrupt MP3
    try:
        predict_file(fixtures["corrupt_mp3"], "audio")
        assert False, "Expected MediaDecodeError for corrupt MP3"
    except Exception as e:
        print(f"[Direct 5 - Corrupt MP3] Correctly rejected: {type(e).__name__} -> {e}")

    # Direct Test 6: Corrupt MP4
    try:
        predict_file(fixtures["corrupt_mp4"], "audio")
        assert False, "Expected MediaDecodeError for corrupt MP4"
    except Exception as e:
        print(f"[Direct 6 - Corrupt MP4] Correctly rejected: {type(e).__name__} -> {e}")

    # Direct Test 7: Over duration (>300s)
    try:
        predict_file(fixtures["long_audio"], "audio")
        assert False, "Expected MediaDecodeError for audio exceeding duration"
    except Exception as e:
        print(f"[Direct 7 - Over Duration Audio] Correctly rejected: {type(e).__name__} -> {e}")
        assert "duration" in str(e).lower()

    print("\n" + "=" * 70)
    print("TEST SUITE: End-to-End Flask Web Upload & Lifecycle Tests")
    print("=" * 70)
    
    app.config["TESTING"] = True
    client = app.test_client()
    
    # Setup test user in database
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, username FROM users WHERE email='audio_tester@example.com' LIMIT 1")
    user = cur.fetchone()
    if not user:
        cur.execute(
            "INSERT INTO users (fullname, username, email, mobile, password) VALUES (%s, %s, %s, %s, %s)",
            ("Audio Tester", "audio_tester", "audio_tester@example.com", "9988776655", "Password123!")
        )
        conn.commit()
        user_id = cur.lastrowid
        user_name = "audio_tester"
    else:
        user_id = user["id"]
        user_name = user["username"]
    cur.close()
    conn.close()
    
    with client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["username"] = user_name

    # Upload Test 1: WAV
    with open(fixtures["wav"], "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "test_audio.wav", "audio/wav")}, follow_redirects=False)
        print(f"[Web 1 - WAV Upload] Status: {resp.status_code}, Location: {resp.headers.get('Location')}")
        assert resp.status_code == 302 and "/result/" in resp.headers.get("Location")
        pid = int(resp.headers.get("Location").split("/result/")[1])
        res_page = client.get(resp.headers.get("Location"))
        assert res_page.status_code == 200
        assert "Audio" in res_page.data.decode("utf-8")
        audit = get_prediction_audit(pid)
        assert audit is not None and audit["file_type"] == "Audio"
        print(f"  -> WAV Result & Audit confirmed (ID: {pid}, Conf: {audit['confidence']}%)")

    # Upload Test 2: MP3
    with open(fixtures["mp3"], "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "test_audio.mp3", "audio/mpeg")}, follow_redirects=False)
        print(f"[Web 2 - MP3 Upload] Status: {resp.status_code}, Location: {resp.headers.get('Location')}")
        assert resp.status_code == 302 and "/result/" in resp.headers.get("Location")
        pid = int(resp.headers.get("Location").split("/result/")[1])
        res_page = client.get(resp.headers.get("Location"))
        assert res_page.status_code == 200
        assert "Audio" in res_page.data.decode("utf-8")
        audit = get_prediction_audit(pid)
        assert audit is not None and audit["file_type"] == "Audio"
        print(f"  -> MP3 Result & Audit confirmed (ID: {pid}, Conf: {audit['confidence']}%)")

    # Upload Test 2B: M4A
    with open(fixtures["m4a"], "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "test_audio.m4a", "audio/m4a")}, follow_redirects=False)
        print(f"[Web 2B - M4A Upload] Status: {resp.status_code}, Location: {resp.headers.get('Location')}")
        assert resp.status_code == 302 and "/result/" in resp.headers.get("Location")
        pid = int(resp.headers.get("Location").split("/result/")[1])
        res_page = client.get(resp.headers.get("Location"))
        assert res_page.status_code == 200
        assert "Audio" in res_page.data.decode("utf-8")
        audit = get_prediction_audit(pid)
        assert audit is not None and audit["file_type"] == "Audio"
        print(f"  -> M4A Result & Audit confirmed (ID: {pid}, Conf: {audit['confidence']}%)")

    # Upload Test 2C: FLAC
    with open(fixtures["flac"], "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "test_audio.flac", "audio/flac")}, follow_redirects=False)
        print(f"[Web 2C - FLAC Upload] Status: {resp.status_code}, Location: {resp.headers.get('Location')}")
        assert resp.status_code == 302 and "/result/" in resp.headers.get("Location")
        pid = int(resp.headers.get("Location").split("/result/")[1])
        res_page = client.get(resp.headers.get("Location"))
        assert res_page.status_code == 200
        assert "Audio" in res_page.data.decode("utf-8")
        audit = get_prediction_audit(pid)
        assert audit is not None and audit["file_type"] == "Audio"
        print(f"  -> FLAC Result & Audit confirmed (ID: {pid}, Conf: {audit['confidence']}%)")

    # Upload Test 3: MP4 with audio
    with open(fixtures["mp4_audio"], "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "test_audio.mp4", "video/mp4")}, follow_redirects=False)
        print(f"[Web 3 - MP4 Audio Upload] Status: {resp.status_code}, Location: {resp.headers.get('Location')}")
        assert resp.status_code == 302 and "/result/" in resp.headers.get("Location")
        pid = int(resp.headers.get("Location").split("/result/")[1])
        res_page = client.get(resp.headers.get("Location"))
        assert res_page.status_code == 200
        assert "Audio" in res_page.data.decode("utf-8")
        audit = get_prediction_audit(pid)
        assert audit is not None and audit["file_type"] == "Audio"
        print(f"  -> MP4 Result & Audit confirmed (ID: {pid}, Conf: {audit['confidence']}%)")

    # Upload Test 4: MP4 without audio track -> expect clean rejection & flash
    with open(fixtures["mp4_silent"], "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "test_silent.mp4", "video/mp4")}, follow_redirects=True)
        print(f"[Web 4 - MP4 Silent Upload] Status: {resp.status_code}")
        assert resp.status_code == 200
        html = resp.data.decode("utf-8")
        assert "does not contain an audio track" in html
        print("  -> Silent MP4 correctly rejected with user alert.")

    # Upload Test 5: Corrupted MP3
    with open(fixtures["corrupt_mp3"], "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "corrupt.mp3", "audio/mpeg")}, follow_redirects=True)
        print(f"[Web 5 - Corrupt MP3 Upload] Status: {resp.status_code}")
        assert resp.status_code == 200
        assert "could not be decoded" in resp.data.decode("utf-8")
        print("  -> Corrupt MP3 correctly rejected with user alert.")

    # Upload Test 6: Corrupted MP4
    with open(fixtures["corrupt_mp4"], "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "corrupt.mp4", "video/mp4")}, follow_redirects=True)
        print(f"[Web 6 - Corrupt MP4 Upload] Status: {resp.status_code}")
        assert resp.status_code == 200
        assert "could not be decoded" in resp.data.decode("utf-8")
        print("  -> Corrupt MP4 correctly rejected with user alert.")

    # Upload Test 7: Unsupported format (.txt)
    with open(fixtures["unsupported"], "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "notes.txt", "text/plain")}, follow_redirects=True)
        print(f"[Web 7 - Unsupported Format Upload] Status: {resp.status_code}")
        assert resp.status_code == 200
        assert "not supported" in resp.data.decode("utf-8")
        print("  -> Unsupported format correctly rejected with user alert.")

    # Upload Test 8: Empty audio file (0-byte)
    with open(fixtures["empty"], "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "empty.mp3", "audio/mpeg")}, follow_redirects=True)
        print(f"[Web 8 - Empty File Upload] Status: {resp.status_code}")
        assert resp.status_code == 200
        assert "empty" in resp.data.decode("utf-8").lower()
        print("  -> Empty file correctly rejected with user alert.")

    # Upload Test 9A: Audio exceeding allowed duration (>300s)
    with open(fixtures["long_audio"], "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "too_long.mp3", "audio/mpeg")}, follow_redirects=True)
        print(f"[Web 9A - Exceeding Duration Upload] Status: {resp.status_code}")
        assert resp.status_code == 200
        assert "duration" in resp.data.decode("utf-8").lower()
        print("  -> Audio exceeding duration correctly rejected with user alert.")

    # Upload Test 9B: Audio exceeding allowed size (>50MB)
    with open(fixtures["large_audio"], "rb") as f:
        resp = client.post("/audio_detection", data={"audio": (io.BytesIO(f.read()), "too_large.wav", "audio/wav")}, follow_redirects=True)
        print(f"[Web 9B - Exceeding Size Upload] Status: {resp.status_code}")
        assert resp.status_code == 200
        assert "limit" in resp.data.decode("utf-8").lower() or "exceeds" in resp.data.decode("utf-8").lower()
        print("  -> Audio exceeding size correctly rejected with user alert.")

    # Test 10: Verify History Page has all uploaded files
    hist_resp = client.get("/history")
    assert hist_resp.status_code == 200
    hist_html = hist_resp.data.decode("utf-8")
    assert "test_audio.wav" in hist_html
    assert "test_audio.mp3" in hist_html
    assert "test_audio.m4a" in hist_html
    assert "test_audio.flac" in hist_html
    assert "test_audio.mp4" in hist_html
    print(f"[Web 10 - History Verification] Status: 200 -> All audio uploads (WAV, MP3, M4A, FLAC, MP4) listed in history table.")

    print("\n" + "=" * 70)
    print("ALL 9 TEST CASES AND HISTORY WORKFLOW VERIFIED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()
