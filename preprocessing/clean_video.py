import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

def extract_frames_from_video(
    video_path: str,
    num_frames: int = 10,
    image_size: tuple = (224, 224),
    strict: bool = False,
):
    """Extracts num_frames frames uniformly sampled from the video with aspect-ratio center-crop."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    if not cap.isOpened():
        cap.release()
        if strict:
            raise ValueError("The video could not be opened.")
        return np.full((num_frames, image_size[0], image_size[1], 3), -1.0, dtype=np.float32)
        
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = num_frames

        indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                if strict:
                    raise ValueError("A sampled video frame could not be decoded.")
                frames.append(np.full((image_size[0], image_size[1], 3), -1.0, dtype=np.float32))
                continue
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            if h != w:
                min_dim = min(h, w)
                top = (h - min_dim) // 2
                left = (w - min_dim) // 2
                frame = frame[top:top + min_dim, left:left + min_dim]
            frame = cv2.resize(frame, image_size)
            frame = tf.keras.applications.mobilenet_v2.preprocess_input(
                frame.astype(np.float32)
            )
            frames.append(frame)
    finally:
        cap.release()

    return np.array(frames, dtype=np.float32)


def extract_video_windows(
    video_path: str,
    num_windows: int = 3,
    window_size: int = 5,
    image_size: tuple = (224, 224),
    strict: bool = False,
):
    """
    Extracts multiple temporal windows of frames uniformly sampled across video segments.
    Each window contains window_size frames (5 frames, matching the Bi-GRU temporal depth).
    Returns an array of shape (num_windows, window_size, height, width, 3).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        if strict:
            raise ValueError("The video could not be opened.")
        return np.zeros((1, window_size, image_size[0], image_size[1], 3), dtype=np.float32)

    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = window_size

        actual_windows = num_windows if total_frames >= (num_windows * window_size) else 1
        segment_length = max(1.0, float(total_frames) / actual_windows)
        all_windows = []

        for w_idx in range(actual_windows):
            seg_start = int(w_idx * segment_length)
            seg_end = int(min(total_frames - 1, (w_idx + 1) * segment_length))
            indices = np.linspace(seg_start, max(seg_start, seg_end), window_size, dtype=int)

            window_frames = []
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    if strict and len(window_frames) == 0:
                        raise ValueError("A sampled video frame could not be decoded.")
                    window_frames.append(np.zeros((image_size[0], image_size[1], 3), dtype=np.float32))
                    continue

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w_dim, _ = frame.shape
                if h != w_dim:
                    min_dim = min(h, w_dim)
                    top = (h - min_dim) // 2
                    left = (w_dim - min_dim) // 2
                    frame = frame[top:top + min_dim, left:left + min_dim]
                frame = cv2.resize(frame, image_size)
                frame = tf.keras.applications.mobilenet_v2.preprocess_input(
                    frame.astype(np.float32)
                )
                window_frames.append(frame)

            all_windows.append(window_frames)
    finally:
        cap.release()

    return np.array(all_windows, dtype=np.float32)


def video_generator(directory: str, num_frames: int = 10, image_size: tuple = (224, 224)):
    """Generator yielding video frames and label."""
    classes = ['fake', 'real']
    for label, class_name in enumerate(classes):
        class_dir = Path(directory) / class_name
        if not class_dir.exists():
            continue
            
        for video_path in class_dir.rglob("*"):
            if not video_path.is_file():
                continue
            
            ext = video_path.suffix.lower()
            if ext not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
                continue
                
            frames = extract_frames_from_video(str(video_path), num_frames, image_size)
            # Binary label: fake=0, real=1 based on index in classes
            yield frames, np.array([label], dtype=np.float32)


def get_video_dataset(directory: str | Path, batch_size: int = 4, num_frames: int = 10, image_size: tuple = (224, 224), shuffle: bool = True):
    """
    Creates a tf.data.Dataset for videos, streaming frames on the fly.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory {dir_path} does not exist.")

    dataset = tf.data.Dataset.from_generator(
        lambda: video_generator(str(dir_path), num_frames, image_size),
        output_signature=(
            tf.TensorSpec(shape=(num_frames, image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.float32)
        )
    )
    
    if shuffle:
        # Buffer size can't be too large due to RAM limits (videos are large)
        dataset = dataset.shuffle(buffer_size=50)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
