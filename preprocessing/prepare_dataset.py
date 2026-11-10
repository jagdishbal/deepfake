"""Prepare downloaded raw datasets for the Deepfake-Analysis project.

The script scans the dataset/raw tree, detects common image/audio/video files,
classifies them as real or fake based on folder names or labels, copies them into
train/validation/test splits, and writes summary reports.
"""

from __future__ import annotations

import csv
import random
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".aac"}


def _iter_candidate_files(raw_dir: Path) -> List[Path]:
    """Collect supported files recursively from the raw dataset directory."""
    if not raw_dir.exists():
        return []

    files: List[Path] = []
    for path in raw_dir.rglob('*'):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS:
            files.append(path)
    return sorted(files)


def _infer_label(path: Path, media_type: str) -> str:
    """Infer the label from the nearest class folder, respecting AVCeleb modality labels."""
    for parent in path.parents:
        if parent.name.casefold() == "raw":
            break
        name = re.sub(r"[^a-z0-9]+", "", parent.name.casefold())

        fake_video = "fakevideo" in name
        real_video = "realvideo" in name
        fake_audio = "fakeaudio" in name
        real_audio = "realaudio" in name
        if fake_video or real_video or fake_audio or real_audio:
            if media_type == "audio":
                fake, real = fake_audio, real_audio
            else:
                fake, real = fake_video, real_video
            if fake != real:
                return "fake" if fake else "real"
            continue

        tokens = set(re.findall(r"[a-z0-9]+", parent.name.casefold()))
        fake = bool(tokens & {"fake", "deepfake", "deepfakes", "manipulated", "forged", "spoof", "attack", "face2face", "faceshifter", "faceswap"})
        real = bool(tokens & {"real", "authentic", "original", "bonafide", "legit"})
        if fake != real:
            return "fake" if fake else "real"

    return "unknown"


def _is_readable_file(path: Path) -> bool:
    """Best-effort validation for common media files."""
    try:
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            from PIL import Image

            with Image.open(path) as image:
                image.verify()
            return True
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            import cv2

            cap = cv2.VideoCapture(str(path))
            success = bool(cap.isOpened())
            cap.release()
            return success
        if path.suffix.lower() in AUDIO_EXTENSIONS:
            import wave

            with wave.open(str(path), "rb") as audio_file:
                audio_file.getnframes()
            return True
    except Exception:
        return False
    return True


def _build_target_dir(base_dir: Path, media_type: str, split: str, label: str) -> Path:
    """Create the destination directory for a prepared file."""
    target_dir = base_dir / media_type / split / label
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _copy_with_unique_name(source: Path, destination_dir: Path) -> Path:
    """Hardlink or copy a file to a directory, renaming it if a conflict exists."""
    destination = destination_dir / source.name
    if destination.exists():
        stem = source.stem
        suffix = source.suffix
        counter = 1
        while destination.exists():
            destination = destination_dir / f"{stem}_{counter}{suffix}"
            counter += 1
    try:
        import os
        os.link(str(source), str(destination))
    except Exception:
        shutil.copy2(source, destination)
    return destination


def import_external_dataset(source_dir: Path | str, target_root: Path | str | None = None) -> List[Path]:
    """Copy media files from an external directory into the project's raw dataset tree."""
    if target_root is None:
        target_root = Path(__file__).resolve().parents[1]
    target_root = Path(target_root)

    source_dir = Path(source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    raw_target_dir = target_root / "dataset" / "raw"
    raw_target_dir.mkdir(parents=True, exist_ok=True)

    copied_files: List[Path] = []
    source_name_is_label = source_dir.name.casefold() in {
        "fake",
        "deepfake",
        "deepfakes",
        "manipulated",
        "forged",
        "spoof",
        "attack",
        "real",
        "authentic",
        "original",
        "bonafide",
        "legit",
    }
    for path in tqdm(sorted(source_dir.rglob('*')), desc="Importing external dataset"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS | AUDIO_EXTENSIONS:
            continue

        relative_path = path.relative_to(source_dir)
        if source_name_is_label:
            relative_path = Path(source_dir.name) / relative_path
        destination_dir = raw_target_dir / relative_path.parent
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = _copy_with_unique_name(path, destination_dir)
        copied_files.append(destination)

    return copied_files


def _split_records(records: List[Tuple[Path, str, str]]) -> Dict[str, List[Tuple[Path, str, str]]]:
    """Split records into train/validation/test while handling small datasets."""
    if not records:
        return {"train": [], "validation": [], "test": []}

    if len(records) <= 2:
        return {
            "train": records[:1],
            "validation": records[1:2],
            "test": [],
        }

    try:
        train_items, remainder = train_test_split(records, train_size=0.7, random_state=42, shuffle=True)
        validation_items, test_items = train_test_split(remainder, test_size=0.5, random_state=42, shuffle=True)
        return {
            "train": list(train_items),
            "validation": list(validation_items),
            "test": list(test_items),
        }
    except ValueError:
        total = len(records)
        train_count = max(1, int(round(total * 0.7)))
        validation_count = max(1, int(round(total * 0.15))) if total > 1 else 0
        test_count = total - train_count - validation_count
        if test_count < 0:
            validation_count = max(0, total - train_count)
            test_count = total - train_count - validation_count
        if test_count < 0:
            test_count = 0
        return {
            "train": records[:train_count],
            "validation": records[train_count:train_count + validation_count],
            "test": records[train_count + validation_count:train_count + validation_count + test_count],
        }


def _source_group(path: Path) -> str:
    """Group extracted frames/audio/video that share a source identity or clip folder."""
    for parent in path.parents:
        if parent.name.casefold() == "raw":
            break
        if re.fullmatch(r"(?:id\d+|\d+_\d+)", parent.name, re.IGNORECASE):
            return parent.name.casefold()
    frame_match = re.fullmatch(r"(.+)_frame\d+", path.stem, re.IGNORECASE)
    if frame_match:
        return frame_match.group(1).casefold()
    return str(path.resolve()).casefold()


def _split_grouped_records(
    records: List[Tuple[Path, str, str]],
) -> Dict[str, List[Tuple[Path, str, str]]]:
    """Split large modality datasets by source group to prevent train/test leakage."""
    if not records:
        return {"train": [], "validation": [], "test": []}

    labels = [record[2] for record in records]
    groups = [_source_group(record[0]) for record in records]
    label_totals = {label: labels.count(label) for label in ("fake", "real")}
    if len(set(groups)) < 3:
        raise ValueError("At least three independent source groups are required for a grouped split.")

    best_split = None
    best_score = float("inf")
    for seed in range(42, 142):
        first_split = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
        train_indices, remainder_indices = next(
            first_split.split(records, labels, groups)
        )
        remainder_records = [records[index] for index in remainder_indices]
        remainder_labels = [labels[index] for index in remainder_indices]
        remainder_groups = [groups[index] for index in remainder_indices]
        second_split = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        validation_relative, test_relative = next(
            second_split.split(remainder_records, remainder_labels, remainder_groups)
        )
        validation_indices = remainder_indices[validation_relative]
        test_indices = remainder_indices[test_relative]
        split_indices = {
            "train": train_indices,
            "validation": validation_indices,
            "test": test_indices,
        }

        if any(
            {labels[index] for index in indices} != {"fake", "real"}
            for indices in split_indices.values()
        ):
            continue

        score = 0.0
        for label in ("fake", "real"):
            label_total = label_totals[label]
            if not label_total:
                continue
            for split, target_fraction in (
                ("train", 0.7),
                ("validation", 0.15),
                ("test", 0.15),
            ):
                actual_fraction = sum(
                    labels[index] == label for index in split_indices[split]
                ) / label_total
                score += abs(actual_fraction - target_fraction)

        if score < best_score:
            best_score = score
            best_split = {
                split: [records[index] for index in indices]
                for split, indices in split_indices.items()
            }

    if best_split is None:
        raise ValueError(
            "Could not create grouped train/validation/test splits containing both labels."
        )
    return best_split


def process_file(path: Path):
    if not _is_readable_file(path):
        return None
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        media_type = "image"
    elif suffix in VIDEO_EXTENSIONS:
        media_type = "video"
    else:
        media_type = "audio"

    label = _infer_label(path, media_type)
    if label == "unknown":
        return None
    if label not in {"real", "fake"}:
        return None
    return (path, media_type, label)


def prepare_dataset(project_root: Path | None = None) -> Dict[str, int]:
    """Copy raw datasets into train/validation/test folders and write reports."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "dataset" / "raw"
    base_output_dir = project_root / "dataset"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    media_types = {
        "image": IMAGE_EXTENSIONS,
        "video": VIDEO_EXTENSIONS,
        "audio": AUDIO_EXTENSIONS,
    }

    for media_type, extensions in media_types.items():
        for split in ["train", "validation", "test"]:
            for label in ["real", "fake"]:
                (base_output_dir / media_type / split / label).mkdir(parents=True, exist_ok=True)

    samples: List[Tuple[Path, str, str]] = []
    skipped = 0

    candidate_files = _iter_candidate_files(raw_dir)
    
    if len(candidate_files) < 100:
        results = list(
            tqdm(
                map(process_file, candidate_files),
                total=len(candidate_files),
                desc="Scanning raw dataset",
            )
        )
    else:
        import concurrent.futures

        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
            results = list(
                tqdm(
                    executor.map(process_file, candidate_files),
                    total=len(candidate_files),
                    desc="Scanning raw dataset",
                )
            )
        
    for res in results:
        if res is None:
            skipped += 1
        else:
            samples.append(res)

    random.seed(42)
    random.shuffle(samples)

    grouped: Dict[str, List[Tuple[Path, str, str]]] = {"image": [], "video": [], "audio": []}
    for path, media_type, label in samples:
        grouped[media_type].append((path, media_type, label))

    stats = {
        "total_images": 0,
        "total_videos": 0,
        "total_audio": 0,
        "real_images": 0,
        "fake_images": 0,
        "real_videos": 0,
        "fake_videos": 0,
        "real_audio": 0,
        "fake_audio": 0,
        "train_count": 0,
        "validation_count": 0,
        "test_count": 0,
        "skipped_files": 0,
    }

    for media_type in ["image", "video", "audio"]:
        records = grouped[media_type]
        if not records:
            continue

        media_labels = [label for _, _, label in records if label in {"real", "fake"}]
        if len(media_labels) == 0:
            continue

        real_records = [record for record in records if record[2] == "real"]
        fake_records = [record for record in records if record[2] == "fake"]

        if real_records and fake_records and len(records) >= 30:
            split_records = _split_grouped_records(records)
        elif len(real_records) and len(fake_records):
            real_split = _split_records(real_records)
            fake_split = _split_records(fake_records)
            split_records = {
                "train": real_split["train"] + fake_split["train"],
                "validation": real_split["validation"] + fake_split["validation"],
                "test": real_split["test"] + fake_split["test"],
            }
        else:
            split_records = {
                "train": records,
                "validation": [],
                "test": [],
            }

        for split, split_items in split_records.items():
            for source, _, label in split_items:
                target_dir = _build_target_dir(base_output_dir, media_type, split, label)
                destination = _copy_with_unique_name(source, target_dir)
                if media_type == "image":
                    stats["total_images"] += 1
                    if label == "real":
                        stats["real_images"] += 1
                    else:
                        stats["fake_images"] += 1
                elif media_type == "video":
                    stats["total_videos"] += 1
                    if label == "real":
                        stats["real_videos"] += 1
                    else:
                        stats["fake_videos"] += 1
                else:
                    stats["total_audio"] += 1
                    if label == "real":
                        stats["real_audio"] += 1
                    else:
                        stats["fake_audio"] += 1

                stats[f"{split}_count"] += 1

    stats["skipped_files"] = skipped

    summary_lines = [
        "Dataset Summary",
        "===============",
        f"Total Images: {stats['total_images']}",
        f"Total Videos: {stats['total_videos']}",
        f"Total Audio Files: {stats['total_audio']}",
        f"Real Images: {stats['real_images']}",
        f"Fake Images: {stats['fake_images']}",
        f"Real Videos: {stats['real_videos']}",
        f"Fake Videos: {stats['fake_videos']}",
        f"Real Audio: {stats['real_audio']}",
        f"Fake Audio: {stats['fake_audio']}",
        f"Train Count: {stats['train_count']}",
        f"Validation Count: {stats['validation_count']}",
        f"Test Count: {stats['test_count']}",
        f"Skipped Files: {stats['skipped_files']}",
    ]
    (reports_dir / "dataset_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    with (reports_dir / "dataset_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "value",
            ],
        )
        writer.writeheader()
        for key, value in stats.items():
            writer.writerow({"metric": key, "value": value})

    return stats


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    prepare_dataset(project_root=project_root)
    print("Dataset preparation completed successfully.")
