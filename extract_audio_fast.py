import os
from pathlib import Path
from moviepy import VideoFileClip

def extract_and_split():
    video_dir = Path("dataset/video")
    audio_dir = Path("dataset/audio")
    
    for split in ['train', 'validation', 'test']:
        for cls in ['real', 'fake']:
            src = video_dir / split / cls
            dst = audio_dir / split / cls
            dst.mkdir(parents=True, exist_ok=True)
            
            if not src.exists():
                continue
                
            all_mp4s = list(src.rglob("*.mp4"))
            count = 0
            for mp4_file in all_mp4s:
                try:
                    wav_path = dst / f"{mp4_file.stem}.wav"
                    if not wav_path.exists():
                        video = VideoFileClip(str(mp4_file))
                        if video.audio:
                            video.audio.write_audiofile(str(wav_path), logger=None)
                            count += 1
                        video.close()
                except Exception as e:
                    print(f"Failed to extract {mp4_file}: {e}")
            print(f"Extracted {count} audio files for {cls} in {split}.")

if __name__ == "__main__":
    extract_and_split()
