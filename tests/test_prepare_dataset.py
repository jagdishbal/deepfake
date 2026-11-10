import importlib.util
import wave
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "preprocessing" / "prepare_dataset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("prepare_dataset", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_prepare_dataset_creates_split_directories(tmp_path):
    module = load_module()
    raw_dir = tmp_path / "dataset" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    image_real_dir = raw_dir / "images" / "real"
    image_fake_dir = raw_dir / "images" / "fake"
    image_real_dir.mkdir(parents=True, exist_ok=True)
    image_fake_dir.mkdir(parents=True, exist_ok=True)

    for index in range(2):
        image = Image.new("RGB", (16, 16), color="red")
        image.save(image_real_dir / f"real_{index}.jpg")
        image = Image.new("RGB", (16, 16), color="blue")
        image.save(image_fake_dir / f"fake_{index}.jpg")

    audio_real_dir = raw_dir / "audio" / "real"
    audio_fake_dir = raw_dir / "audio" / "fake"
    audio_real_dir.mkdir(parents=True, exist_ok=True)
    audio_fake_dir.mkdir(parents=True, exist_ok=True)

    for index, target_dir in enumerate([audio_real_dir, audio_fake_dir]):
        with wave.open(str(target_dir / f"sample_{index}.wav"), "wb") as audio_file:
            audio_file.setnchannels(1)
            audio_file.setsampwidth(2)
            audio_file.setframerate(8000)
            audio_file.writeframes(b"\x00\x01" * 100)

    stats = module.prepare_dataset(project_root=tmp_path)

    assert stats["total_images"] == 4
    assert stats["real_images"] == 2
    assert stats["fake_images"] == 2
    assert stats["total_audio"] == 2
    assert stats["real_audio"] == 1
    assert stats["fake_audio"] == 1
    assert (tmp_path / "dataset" / "image" / "train" / "real").exists()
    assert (tmp_path / "dataset" / "audio" / "test" / "fake").exists()


def test_import_external_dataset_copies_to_raw(tmp_path):
    module = load_module()
    source_dir = tmp_path / "external" / "real"
    source_dir.mkdir(parents=True, exist_ok=True)

    image_path = source_dir / "sample.jpg"
    image = Image.new("RGB", (16, 16), color="green")
    image.save(image_path)

    copied_files = module.import_external_dataset(source_dir=source_dir, target_root=tmp_path)

    assert len(copied_files) == 1
    assert (tmp_path / "dataset" / "raw" / "real" / "sample.jpg").exists()


def test_infer_label_uses_video_or_audio_part_of_fakeavcelebrity_path():
    module = load_module()
    video_fake_audio_real = Path(
        "raw/images/FakeAVCeleb_v1.2/frames/FakeVideo-RealAudio/id00001/frame.jpg"
    )
    video_real_audio_fake = Path(
        "raw/audio/FakeAVCeleb_v1.2/RealVideo-FakeAudio/id00001/sample.wav"
    )

    assert module._infer_label(video_fake_audio_real, "image") == "fake"
    assert module._infer_label(video_fake_audio_real, "audio") == "real"
    assert module._infer_label(video_real_audio_fake, "video") == "real"
    assert module._infer_label(video_real_audio_fake, "audio") == "fake"


def test_infer_label_uses_class_folder_not_filename_and_skips_unknown(tmp_path):
    module = load_module()
    real_file = tmp_path / "raw" / "real" / "sample_fake.jpg"
    unknown_file = tmp_path / "raw" / "unlabeled" / "sample.jpg"

    assert module._infer_label(real_file, "image") == "real"
    assert module._infer_label(unknown_file, "image") == "unknown"


def test_large_dataset_split_keeps_source_groups_together():
    module = load_module()
    records = []
    for index in range(20):
        group = f"id{index:05d}"
        records.append((Path(f"raw/fake/{group}/fake.jpg"), "image", "fake"))
        records.append((Path(f"raw/real/{group}/real.jpg"), "image", "real"))

    split = module._split_grouped_records(records)

    groups_by_split = {
        name: {module._source_group(path) for path, _, _ in split[name]}
        for name in ("train", "validation", "test")
    }
    assert not groups_by_split["train"] & groups_by_split["validation"]
    assert not groups_by_split["train"] & groups_by_split["test"]
    assert not groups_by_split["validation"] & groups_by_split["test"]
    assert all({label for _, _, label in split[name]} == {"fake", "real"} for name in split)
