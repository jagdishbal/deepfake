import os
import cv2
import librosa
from PIL import Image
from pathlib import Path

def validate_modality(modality, extensions, check_func):
    root = Path(f'D:/deepfake_college_project/Deepfake-Analysis/dataset/{modality}')
    print(f"Validating {modality} dataset...")
    corrupt_count = 0
    total_count = 0
    for split in ['train', 'validation', 'test']:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for class_name in ['real', 'fake']:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                continue
            for file_path in class_dir.rglob('*'):
                if file_path.is_file():
                    total_count += 1
                    if file_path.suffix.lower() not in extensions:
                        print(f"Invalid extension, removing: {file_path}")
                        file_path.unlink()
                        corrupt_count += 1
                        continue
                    if not check_func(file_path):
                        print(f"Corrupt file, removing: {file_path}")
                        file_path.unlink()
                        corrupt_count += 1
    print(f"[{modality}] Total: {total_count}, Corrupt/Invalid removed: {corrupt_count}")

def check_image(path):
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except:
        return False

def check_video(path):
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return False
        ret, frame = cap.read()
        cap.release()
        return ret and frame is not None
    except:
        return False

def check_audio(path):
    try:
        _ = librosa.load(path, sr=None, duration=0.1)
        return True
    except:
        return False

if __name__ == '__main__':
    validate_modality('image', {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}, check_image)
    validate_modality('video', {'.mp4', '.avi', '.mov', '.mkv', '.webm'}, check_video)
    validate_modality('audio', {'.wav', '.mp3', '.aac', '.flac', '.ogg', '.m4a'}, check_audio)

