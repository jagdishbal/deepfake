import os
import random
from pathlib import Path
from moviepy import VideoFileClip

def extract_and_split():
    raw_dir = Path("dataset/raw/videos/video_dataset")
    audio_dir = Path("dataset/audio")
    
    random.seed(42)
    
    for cls in ['real', 'fake']:
        src = raw_dir / cls
        if not src.exists():
            print(f"Could not find {src}")
            # Try to search recursively if structure is different
            all_mp4s = list(Path("dataset/raw").rglob(f"*{cls}*.mp4"))
            if not all_mp4s:
                all_mp4s = list(Path("dataset/raw").rglob("*.mp4"))
                all_mp4s = [f for f in all_mp4s if cls in str(f).lower()]
        else:
            all_mp4s = list(src.rglob("*.mp4"))
            
        print(f"Found {len(all_mp4s)} mp4 files for class {cls}.")
        
        # We need a healthy batch (e.g., ~250 per class)
        target_count = min(300, len(all_mp4s))
        selected_mp4s = random.sample(all_mp4s, target_count)
        
        # Calculate splits (70% train, 15% validation, 15% test)
        num_train = int(target_count * 0.70)
        num_val = int(target_count * 0.15)
        
        train_set = selected_mp4s[:num_train]
        val_set = selected_mp4s[num_train:num_train+num_val]
        test_set = selected_mp4s[num_train+num_val:]
        
        splits = {
            'train': train_set,
            'validation': val_set,
            'test': test_set
        }
        
        for split_name, files in splits.items():
            dst = audio_dir / split_name / cls
            dst.mkdir(parents=True, exist_ok=True)
            
            count = 0
            for mp4_file in files:
                try:
                    wav_path = dst / f"{mp4_file.stem}.wav"
                    if not wav_path.exists():
                        video = VideoFileClip(str(mp4_file))
                        if video.audio:
                            video.audio.write_audiofile(str(wav_path), verbose=False, logger=None)
                            count += 1
                        video.close()
                except Exception as e:
                    pass
            print(f"Extracted {count} audio files for {cls} in {split_name}.")

if __name__ == "__main__":
    extract_and_split()
