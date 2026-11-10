import config
from pathlib import Path
from predict import predict_file

print("=== 1. TESTING IMAGE INFERENCE (V2 Model) ===")
test_images = [
    "../ai-generated-8622236_1280.jpg",
    "../OIP.jpg",
    "static/uploads/cf36c9647ce14667b790c93c3595fca9_IMG_20240207_174754.jpg",
    "static/uploads/4d3090911ad442b0ada6c6e82ab9e30c_bicky.jpg",
    "static/uploads/a641da006b5d4ae09a6327318ff24bb6_shadow.jpeg",
]

for img_p in test_images:
    p = Path(img_p)
    if p.exists():
        res = predict_file(p, modality="image")
        pred = res["prediction"]
        conf = res["confidence"] * 100
        prob = res["probability_real"]
        print(f"{p.name:<45} -> {pred:<5} | Conf: {conf:5.1f}% | Prob(Real): {prob:.4f}")
    else:
        print(f"{p.name:<45} -> NOT FOUND")

print("\n=== 2. TESTING VIDEO INFERENCE ===")
vid_files = list(Path("dataset/video/test").rglob("*.mp4"))[:2]
for vf in vid_files:
    res = predict_file(vf, modality="video")
    pred = res["prediction"]
    conf = res["confidence"] * 100
    prob = res["probability_real"]
    print(f"{vf.name:<45} -> {pred:<5} | Conf: {conf:5.1f}% | Prob(Real): {prob:.4f}")

print("\n=== 3. TESTING AUDIO INFERENCE ===")
aud_files = list(Path("dataset/audio/test").rglob("*.wav"))[:2]
for af in aud_files:
    res = predict_file(af, modality="audio")
    pred = res["prediction"]
    conf = res["confidence"] * 100
    prob = res["probability_real"]
    print(f"{af.name:<45} -> {pred:<5} | Conf: {conf:5.1f}% | Prob(Real): {prob:.4f}")
