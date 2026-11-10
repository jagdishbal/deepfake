import config
from pathlib import Path
from predict import predict_file, _load_model, get_threshold
from services.prediction_service import predict_media

print("=" * 80)
print("REGRESSION TESTING FOR VIDEO AND AUDIO MODALITIES")
print("=" * 80)

print("\n--- 1. CHECKING AUDIO & VIDEO CHECKPOINTS & THRESHOLDS ---")
for mod in ["video", "audio"]:
    model_path = Path(f"models/saved_models/{mod}/best_model.h5")
    backup_path = Path(f"models/backups/{mod}_best_model.h5")
    thresh = get_threshold(mod)
    print(f"Modality: {mod:<5} | Threshold: {thresh} | Checkpoint Exists: {model_path.exists()} ({model_path.stat().st_size:,} bytes) | Backup Exists: {backup_path.exists()}")

print("\n--- 2. TESTING VIDEO PREDICTION PIPELINE ---")
vid_files = list(Path("dataset/video/test").rglob("*.mp4"))[:2]
if vid_files:
    for vf in vid_files:
        res = predict_file(vf, modality="video")
        print(f"File: {vf.name:<40} -> Pred: {res['prediction']:<4} | Conf: {res['confidence']*100:.1f}% | Prob(Real): {res['probability_real']:.4f} | Modality: {res['modality']}")
        assert res["prediction"] in {"Real", "Fake"}
        assert res["modality"] == "video"
    print("Video inference pipeline: PASSED")
else:
    print("No test video files found.")

print("\n--- 3. TESTING AUDIO PREDICTION PIPELINE ---")
aud_files = list(Path("dataset/audio/test").rglob("*.wav"))[:2]
if aud_files:
    for af in aud_files:
        res = predict_file(af, modality="audio")
        print(f"File: {af.name:<40} -> Pred: {res['prediction']:<4} | Conf: {res['confidence']*100:.1f}% | Prob(Real): {res['probability_real']:.4f} | Modality: {res['modality']}")
        assert res["prediction"] in {"Real", "Fake"}
        assert res["modality"] == "audio"
    print("Audio inference pipeline: PASSED")
else:
    print("No test audio files found.")

print("\n--- 4. TESTING SERVICES API CONTRACT FOR VIDEO & AUDIO ---")
if vid_files:
    srv_vid = predict_media(vid_files[0], "video")
    print(f"predict_media(video): {srv_vid}")
    assert srv_vid["modality"] == "video"

if aud_files:
    srv_aud = predict_media(aud_files[0], "audio")
    print(f"predict_media(audio): {srv_aud}")
    assert srv_aud["modality"] == "audio"

print("Regression testing for Video and Audio: ALL CHECKS PASSED.")
