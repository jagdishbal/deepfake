import os
import sys
from pathlib import Path
import io

os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from app import app
from predict import predict_file, route_image, get_threshold
from database.database import get_db_connection
from services.audit_service import get_prediction_audit

def run_tests():
    print("=" * 60)
    print("STEP 1: Direct Routing & Prediction Tests")
    print("=" * 60)
    
    # 1. High-Res Image (V3)
    hr_path = Path(r"D:\deepfake_college_project\ai-generated-8622236_1280.jpg")
    assert hr_path.is_file(), f"HR test image missing: {hr_path}"
    route_hr = route_image(hr_path)
    print(f"[HR Routing] Route: {route_hr['route']}, Version: {route_hr['model_version']}, Reason: {route_hr['reason']}")
    assert route_hr["route"] == "v3", f"Expected V3 route, got {route_hr['route']}"
    assert route_hr["threshold"] == 0.3800, f"Expected 0.3800 threshold, got {route_hr['threshold']}"
    
    pred_hr = predict_file(hr_path, "image")
    print(f"[HR Prediction] Result: {pred_hr['prediction']}, Conf: {pred_hr['confidence']:.4f}, Model: {pred_hr['model_used']}")
    assert pred_hr["model_version"] == "V3"
    assert pred_hr["prediction"] == "Fake"
    
    # 2. Low-Res Image (V2)
    lr_path = Path(r"D:\deepfake_college_project\OIP.jpg")
    assert lr_path.is_file(), f"LR test image missing: {lr_path}"
    route_lr = route_image(lr_path)
    print(f"[LR Routing] Route: {route_lr['route']}, Version: {route_lr['model_version']}, Reason: {route_lr['reason']}")
    assert route_lr["route"] == "v2", f"Expected V2 route, got {route_lr['route']}"
    assert route_lr["threshold"] == get_threshold("image", "v2"), f"Expected {get_threshold('image', 'v2')} threshold, got {route_lr['threshold']}"
    
    pred_lr = predict_file(lr_path, "image")
    print(f"[LR Prediction] Result: {pred_lr['prediction']}, Conf: {pred_lr['confidence']:.4f}, Model: {pred_lr['model_used']}")
    assert pred_lr["model_version"] == "V2"
    assert pred_lr["prediction"] == "Fake"

    # 3. Compressed Image (V2 Fallback)
    temp_dir = Path(config.BASE_DIR) / "runtime" / "temp_edge_cases"
    comp_path = temp_dir / "heavily_compressed.jpg"
    if comp_path.is_file():
        route_comp = route_image(comp_path)
        print(f"[Compressed Routing] Route: {route_comp['route']}, Version: {route_comp['model_version']}, Reason: {route_comp['reason']}")
        assert route_comp["route"] == "v2", f"Expected V2 route for compressed image, got {route_comp['route']}"
        assert route_comp["threshold"] == get_threshold("image", "v2")

    print("\n" + "=" * 60)
    print("STEP 2: End-to-End Flask Web Upload & Lifecycle Tests")
    print("=" * 60)
    
    app.config["TESTING"] = True
    client = app.test_client()
    
    # Ensure test user exists in MySQL
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id, username FROM users WHERE email='testuser_workflow@example.com' LIMIT 1")
    user = cur.fetchone()
    if not user:
        cur.execute(
            "INSERT INTO users (fullname, username, email, mobile, password) VALUES (%s, %s, %s, %s, %s)",
            ("Workflow Tester", "workflow_tester", "testuser_workflow@example.com", "9876543210", "Password123!")
        )
        conn.commit()
        user_id = cur.lastrowid
        user_name = "workflow_tester"
    else:
        user_id = user["id"]
        user_name = user["username"]
    cur.close()
    conn.close()
    
    with client.session_transaction() as sess:
        sess["user_id"] = user_id
        sess["username"] = user_name

    # Test 2A: Upload High-Res Image via Form POST
    with open(hr_path, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), "test_hr_upload.jpg")}
        resp = client.post("/image_detection", data=data, content_type="multipart/form-data", follow_redirects=False)
        print(f"[Upload HR] Status: {resp.status_code}, Location: {resp.headers.get('Location')}")
        assert resp.status_code == 302, f"Expected 302 redirect, got {resp.status_code}"
        target_url = resp.headers.get("Location")
        assert "/result/" in target_url, f"Expected redirect to /result/<id>, got {target_url}"
        pred_id = int(target_url.split("/result/")[1])
        
        # Verify Result Page Renders
        res_page = client.get(target_url)
        print(f"[View Result HR] Status: {res_page.status_code}")
        assert res_page.status_code == 200
        html = res_page.data.decode("utf-8")
        assert "Detection Result" in html
        assert "Model V3" in html
        assert "0.38" in html
        
        # Verify Audit Log
        audit = get_prediction_audit(pred_id)
        assert audit is not None
        assert audit["model_version"] == "V3"
        assert audit["threshold"] == 0.3800
        print(f"[Audit Log HR] Model: {audit['model_used']}, Threshold: {audit['threshold']}")

    # Test 2B: Upload Low-Res Image via Form POST
    with open(lr_path, "rb") as f:
        data = {"image": (io.BytesIO(f.read()), "test_lr_upload.jpg")}
        resp = client.post("/image_detection", data=data, content_type="multipart/form-data", follow_redirects=False)
        print(f"[Upload LR] Status: {resp.status_code}, Location: {resp.headers.get('Location')}")
        assert resp.status_code == 302
        target_url = resp.headers.get("Location")
        assert "/result/" in target_url
        pred_id = int(target_url.split("/result/")[1])
        
        # Verify Result Page Renders
        res_page = client.get(target_url)
        print(f"[View Result LR] Status: {res_page.status_code}")
        assert res_page.status_code == 200
        html = res_page.data.decode("utf-8")
        assert "Detection Result" in html
        assert "Model V2" in html
        assert "0.52" in html
        
        # Verify Audit Log
        audit = get_prediction_audit(pred_id)
        assert audit is not None
        assert audit["model_version"] == "V2"
        assert audit["threshold"] == 0.5200
        print(f"[Audit Log LR] Model: {audit['model_used']}, Threshold: {audit['threshold']}")

    # Test 2C: Verify History Page
    hist_page = client.get("/history")
    print(f"[History Page] Status: {hist_page.status_code}")
    assert hist_page.status_code == 200
    hist_html = hist_page.data.decode("utf-8")
    assert "test_hr_upload.jpg" in hist_html
    assert "test_lr_upload.jpg" in hist_html

    print("\n" + "=" * 60)
    print("STEP 3: Invalid & Edge Case Handling")
    print("=" * 60)

    # 3A: Corrupted image bytes
    data = {"image": (io.BytesIO(b"corrupted image content 12345"), "corrupt.jpg")}
    resp = client.post("/image_detection", data=data, content_type="multipart/form-data", follow_redirects=True)
    print(f"[Corrupt Image] Status: {resp.status_code}")
    assert resp.status_code == 200
    assert "The uploaded image is invalid or could not be decoded" in resp.data.decode("utf-8")
    print("  -> Corrupt image correctly rejected with danger flash message.")

    # 3B: Empty 0-byte file
    data = {"image": (io.BytesIO(b""), "empty.jpg")}
    resp = client.post("/image_detection", data=data, content_type="multipart/form-data", follow_redirects=True)
    print(f"[Empty Image] Status: {resp.status_code}")
    assert resp.status_code == 200
    assert "The uploaded file is empty" in resp.data.decode("utf-8")
    print("  -> Empty file correctly rejected with danger flash message.")

    # 3C: Unsupported file format
    data = {"image": (io.BytesIO(b"some text data"), "document.txt")}
    resp = client.post("/image_detection", data=data, content_type="multipart/form-data", follow_redirects=True)
    print(f"[Unsupported Format] Status: {resp.status_code}")
    assert resp.status_code == 200
    assert "That file format is not supported" in resp.data.decode("utf-8")
    print("  -> Unsupported extension correctly rejected with danger flash message.")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! COMPLETE LIFECYCLE VERIFIED.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
