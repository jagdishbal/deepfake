import config
from pathlib import Path
import tempfile
import io
import pytest

from predict import (
    predict_file,
    PredictionError,
    UnsupportedMediaError,
    MediaDecodeError,
)
from app import app

print("=" * 80)
print("TESTING BACKEND PREDICTION ENGINE (8 REQUIRED SCENARIOS)")
print("=" * 80)

# Create a temporary corrupted image
temp_dir = Path("runtime/temp")
temp_dir.mkdir(parents=True, exist_ok=True)
corrupt_file = temp_dir / "corrupted_test_image.jpg"
corrupt_file.write_bytes(b"NOT_A_REAL_IMAGE_BYTES_XYZ123")

# Create an unsupported format file
unsupported_file = temp_dir / "test_unsupported.txt"
unsupported_file.write_text("Hello World, I am a text file.")

scenarios = [
    ("1. Normal Genuine Photograph", Path("static/uploads/cf36c9647ce14667b790c93c3595fca9_IMG_20240207_174754.jpg")),
    ("2. Standard AI-Generated Image", Path("../OIP.jpg")),
    ("3. High-Resolution AI Image", Path("../ai-generated-8622236_1280.jpg")),
    ("4. Portrait Image", Path("static/uploads/4d3090911ad442b0ada6c6e82ab9e30c_bicky.jpg")),
    ("5. Landscape Image", Path("static/images/banner.jpg")),
    ("6. Image Without a Face", Path("static/images/logo.png")),
    ("7. Corrupted Image", corrupt_file),
    ("8. Unsupported File Format", unsupported_file),
]

for label, p in scenarios:
    print(f"\nScenario: {label}")
    print(f"File: {p}")
    try:
        res = predict_file(p, modality="image")
        print(f"  Result: Status=SUCCESS | Pred={res['prediction']} | Conf={res['confidence']*100:.2f}% | Prob(Real)={res['probability_real']:.4f}")
        # Verify structure
        assert "prediction" in res
        assert "confidence" in res
        assert "probability_real" in res
        assert "modality" in res
        assert res["prediction"] in {"Real", "Fake"}
        assert 0.0 <= res["confidence"] <= 1.0
        assert 0.0 <= res["probability_real"] <= 1.0
        print("  Format Verification: PASSED (Keys & types valid)")
    except UnsupportedMediaError as e:
        print(f"  Handled UnsupportedMediaError as expected: {e}")
    except MediaDecodeError as e:
        print(f"  Handled MediaDecodeError as expected: {e}")
    except Exception as e:
        print(f"  FAILED with unexpected error: {type(e).__name__}: {e}")

print("\n" + "=" * 80)
print("TESTING FLASK APPLICATION INTEGRATION VIA TEST CLIENT")
print("=" * 80)

app.config["TESTING"] = True
client = app.test_client()

# Test GET /image_detection when not logged in
res = client.get("/image_detection")
print(f"GET /image_detection (unauthenticated): Status={res.status_code} (Redirect to login={res.location})")

# Test GET /login
res_login = client.get("/login")
print(f"GET /login: Status={res_login.status_code}")

# Test simulated upload flow through prediction_service
from services.prediction_service import predict_media
sample_img = Path("../OIP.jpg")
if sample_img.exists():
    srv_result = predict_media(sample_img, "image")
    print(f"Services API call: {srv_result}")
    assert srv_result["prediction"] in {"Real", "Fake"}
    print("Services API contract: PASSED")

# Clean up temporary test files
corrupt_file.unlink(missing_ok=True)
unsupported_file.unlink(missing_ok=True)
print("\nTemporary test files cleaned up.")
