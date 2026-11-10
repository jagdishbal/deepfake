from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW_ROOT = ROOT / "dataset" / "raw"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
SUPPORTED = {
    "audio": AUDIO_EXTS,
    "images": IMAGE_EXTS,
    "videos": VIDEO_EXTS,
}
EXT_TO_MODALITY = {}
for modality, exts in SUPPORTED.items():
    for ext in exts:
        EXT_TO_MODALITY[ext.lower()] = modality

MODALITY_FOLDER_NAMES = {"audio": "audio", "images": "images", "videos": "videos"}
REAL_HINTS = {"real", "original", "authentic", "bonafide"}
FAKE_HINTS = {"fake", "manipulated", "deepfake", "spoof", "fakes"}


def classify_file(path: Path):
    ext = path.suffix.lower()
    return EXT_TO_MODALITY.get(ext, "unknown")


def ensure_expected_dirs(raw_root: Path):
    for folder in MODALITY_FOLDER_NAMES.values():
        (raw_root / folder).mkdir(parents=True, exist_ok=True)


def relative_subpath(path: Path, base_dir: Path) -> Path:
    return Path(*path.relative_to(base_dir).parts)


def find_misplaced_files(raw_root: Path):
    misplaced = {"audio": [], "images": [], "videos": []}
    counts = {"audio": 0, "images": 0, "videos": 0}
    unknown = []
    real_folders = {"audio": [], "images": [], "videos": []}
    fake_folders = {"audio": [], "images": [], "videos": []}

    for folder_name in MODALITY_FOLDER_NAMES.values():
        folder = raw_root / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_dir():
                lname = path.name.lower()
                if any(h in lname for h in REAL_HINTS):
                    real_folders[folder_name].append(str(path.relative_to(raw_root)))
                if any(h in lname for h in FAKE_HINTS):
                    fake_folders[folder_name].append(str(path.relative_to(raw_root)))
                continue
            if not path.is_file():
                continue
            modality = classify_file(path)
            if modality == "unknown":
                unknown.append(str(path.relative_to(raw_root)))
                continue
            counts[modality] += 1
            if folder_name != MODALITY_FOLDER_NAMES[modality]:
                misplaced[modality].append({
                    "from": str(path.relative_to(raw_root)),
                    "to": str((raw_root / MODALITY_FOLDER_NAMES[modality]) / relative_subpath(path, folder)),
                })

    return {
        "correct": counts,
        "misplaced": misplaced,
        "unknown": sorted(set(unknown)),
        "real_folders": real_folders,
        "fake_folders": fake_folders,
    }


def unique_destination(path: Path):
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    candidate = path
    while candidate.exists():
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        counter += 1
    return candidate


def move_misplaced_files(raw_root: Path, dry_run: bool = True):
    report = find_misplaced_files(raw_root)
    moved = {"audio": [], "images": [], "videos": []}
    skipped = []

    for modality, items in report["misplaced"].items():
        for item in items:
            src = raw_root / item["from"]
            dst_rel = Path(item["to"])
            dst = raw_root / dst_rel
            if not src.exists():
                skipped.append({"src": item["from"], "reason": "missing_source"})
                continue
            if dst.exists() and src.samefile(dst):
                skipped.append({"src": item["from"], "reason": "same_file"})
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dry_run:
                moved[modality].append({"from": item["from"], "to": item["to"]})
            else:
                safe_dest = unique_destination(dst)
                shutil.move(str(src), str(safe_dest))
                moved[modality].append({"from": item["from"], "to": str(safe_dest.relative_to(raw_root))})

    return {"report": report, "moved": moved, "skipped": skipped}


def print_folder_tree(root: Path, depth: int = 2):
    if not root.exists():
        print(f"  [missing] {root}")
        return
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if rel.parts and len(rel.parts) <= depth:
            indent = "  " * len(rel.parts)
            print(f"{indent}{path.name}")


def print_category_summary(label: str, root: Path, report: dict):
    print(f"{label}:")
    print(f"  Correct supported files: {report['correct'].get(label.lower().replace(' dataset', ''), 0)}")
    print(f"  Misplaced files found: {len(report['misplaced'].get(label.lower().replace(' dataset', '').replace(' ', ''), []))}")
    print(f"  Files moved: {len(report['misplaced'].get(label.lower().replace(' dataset', '').replace(' ', ''), []))}")
    print(f"  Unknown files skipped: {len(report['unknown'])}")
    print()


def verify_raw_folders(raw_root: Path):
    issues = []
    for folder_name in MODALITY_FOLDER_NAMES.values():
        folder = raw_root / folder_name
        if not folder.exists():
            continue
        allowed = set(SUPPORTED[folder_name])
        for file_path in sorted(folder.rglob("*")):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in allowed:
                    continue
                if ext == "":
                    continue
                if ext in EXT_TO_MODALITY:
                    issues.append(f"{file_path.relative_to(raw_root)} is in {folder_name} but should be in {EXT_TO_MODALITY[ext]}")
    return issues


def main():
    parser = argparse.ArgumentParser(description="Inspect and correct raw deepfake dataset modality placement.")
    parser.add_argument("--apply", action="store_true", help="Move misplaced files to the correct raw folders.")
    parser.add_argument("--verify", action="store_true", help="Verify each raw folder contains only the expected modality and dataset folders.")
    args = parser.parse_args()

    ensure_expected_dirs(RAW_ROOT)

    if args.verify:
        issues = verify_raw_folders(RAW_ROOT)
        if not issues:
            print("Verification passed: each raw folder contains only the expected modality file types and dataset folders.")
        else:
            print("Verification issues found:")
            for issue in issues:
                print(f" - {issue}")
        return

    report = find_misplaced_files(RAW_ROOT)
    print("DEEPFAKE RAW DATASET INVENTORY")
    print(f"Scanning: {RAW_ROOT}\n")
    print("AUDIO:")
    print(f"  Correct audio files: {report['correct']['audio']}")
    print(f"  Misplaced files found: {len(report['misplaced']['audio'])}")
    print(f"  Files moved: {len(report['misplaced']['audio']) if args.apply else 0}")
    print(f"  Possible REAL folders: {len(report['real_folders']['audio'])}")
    print(f"  Possible FAKE folders: {len(report['fake_folders']['audio'])}")
    print("")

    print("IMAGES:")
    print(f"  Correct image files: {report['correct']['images']}")
    print(f"  Misplaced files found: {len(report['misplaced']['images'])}")
    print(f"  Files moved: {len(report['misplaced']['images']) if args.apply else 0}")
    print(f"  Possible REAL folders: {len(report['real_folders']['images'])}")
    print(f"  Possible FAKE folders: {len(report['fake_folders']['images'])}")
    print("")

    print("VIDEOS:")
    print(f"  Correct video files: {report['correct']['videos']}")
    print(f"  Misplaced files found: {len(report['misplaced']['videos'])}")
    print(f"  Files moved: {len(report['misplaced']['videos']) if args.apply else 0}")
    print(f"  Possible REAL folders: {len(report['real_folders']['videos'])}")
    print(f"  Possible FAKE folders: {len(report['fake_folders']['videos'])}")
    print("")

    total_files = sum(report['correct'].values()) + sum(len(v) for v in report['misplaced'].values())
    print(f"Total files scanned: {total_files}")
    print(f"Total files moved: {sum(len(v) for v in move_misplaced_files(RAW_ROOT, dry_run=True)['moved'].values()) if args.apply else 0}")
    print(f"Total files skipped: {len(report['unknown'])}")
    print(f"Unknown file types: {report['unknown'][:10]}")

    if args.apply:
        result = move_misplaced_files(RAW_ROOT, dry_run=False)
        print("\nFiles moved:")
        for modality, items in result["moved"].items():
            if items:
                print(f"  {modality.upper()}:")
                for item in items:
                    print(f"    {item['from']} -> {item['to']}")
        if result["skipped"]:
            print("\nSkipped entries:")
            for item in result["skipped"]:
                print(f"  {item}")

    else:
        print("\nPlanned moves (dry run):")
        planned = move_misplaced_files(RAW_ROOT, dry_run=True)
        for modality, items in planned["moved"].items():
            if items:
                print(f"  {modality.upper()}:")
                for item in items:
                    print(f"    {item['from']} -> {item['to']}")
        if report["unknown"]:
            print("\nUnknown file types left in place:")
            for item in report["unknown"][:20]:
                print(f"  {item}")


if __name__ == "__main__":
    main()
