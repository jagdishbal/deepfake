import sys
from pathlib import Path

# Need to set python path to import predict
sys.path.append(str(Path(__file__).parent))
from predict import predict_single_file

test_files = [
    # Validation Dataset Images
    ("dataset/image/validation/real/00002_frame0.jpg", "Real (Validation Dataset)"),
    ("dataset/image/validation/fake/00001_id05631_wavtolip_frame0.jpg", "Fake (Validation Dataset)"),
    
    # User's Own Uploads
    ("static/uploads/cf36c9647ce14667b790c93c3595fca9_IMG_20240207_174754.jpg", "User's Real Photo"),
    ("static/uploads/78a59a46cd7f4a3481cef176cfe21058_OIP_1.jpg", "User's AI Photo (OIP_1)")
]

print("=== DEEPFAKE DETECTOR MODEL TEST ===")
for file_path, description in test_files:
    if Path(file_path).exists():
        try:
            result = predict_single_file(file_path)
            pred = result['prediction']
            conf = float(result['confidence']) * 100
            print(f"\n[Testing {description}]")
            print(f"File: {Path(file_path).name}")
            print(f"Result: {pred} ({conf:.2f}% confidence)")
        except Exception as e:
            print(f"Error on {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")
