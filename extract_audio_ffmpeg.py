import os
import subprocess
from pathlib import Path
from imageio_ffmpeg import get_ffmpeg_exe

def extract_and_split():
    ffmpeg_exe = get_ffmpeg_exe()
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
                        # Extract audio using ffmpeg directly
                        cmd = [ffmpeg_exe, "-y", "-i", str(mp4_file), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav_path)]
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                        count += 1
                except Exception as e:
                    print(f"Failed to extract {mp4_file}: {e}")
            print(f"Extracted {count} audio files for {cls} in {split}.")

if __name__ == "__main__":
    extract_and_split()

