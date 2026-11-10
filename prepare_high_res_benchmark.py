import hashlib
import json
import shutil
from pathlib import Path
from PIL import Image

print("=" * 80)
print("PREPARING HIGH-RESOLUTION BENCHMARK DATASET")
print("=" * 80)

base_dir = Path("dataset/high_res_benchmark")
real_dir = base_dir / "real"
fake_dir = base_dir / "fake"

real_dir.mkdir(parents=True, exist_ok=True)
fake_dir.mkdir(parents=True, exist_ok=True)

# Sources for genuine high-res photographs:
# 1. High-resolution phone camera photos from static/uploads (IMG_..., bicky.jpg, 1000032869.jpg, WhatsApp photos >= 1024px)
# 2. High-resolution synthetic AI generations (ai-generated-8622236, ai-generated-8700575, myphoto.edit2, etc.)

candidate_reals = [
    Path("static/uploads/cf36c9647ce14667b790c93c3595fca9_IMG_20240207_174754.jpg"),
    Path("static/uploads/4d3090911ad442b0ada6c6e82ab9e30c_bicky.jpg"),
    Path("static/uploads/8c4b44dafb2743f78085ba70015ac7c3_1000032869.jpg"),
    Path("static/uploads/06395a0d1bff4598ba04a89f4cb50b53_WhatsApp_Image_2026-10-22_at_9.42.38_AM_2.jpeg"),
    Path("static/uploads/0f9b932e68824edfbb8d31ad18125748_WhatsApp_Image_2026-10-22_at_9.42.38_AM.jpeg"),
    Path("static/uploads/b210ba2396964e59a52adcc8a62f2f65_WhatsApp_Image_2026-10-22_at_9.42.38_AM_1.jpeg"),
    Path("static/uploads/219bd5e2e35a461682c4f0edb6827886_bicky_-_Copy.jpg"),
    Path("static/uploads/2e694f64c0b04b0c99a128e780a8f7e2_bicky_1.jpg"),
    Path("static/uploads/096a41120e054a1faee9cdef7a0ea8aa_Screenshot_20240622_181714.jpg"),
    Path("static/uploads/3087aa6685cb449f9504594f0536dece_Screenshot_20240910_094638.jpg"),
]

candidate_fakes = [
    Path("../ai-generated-8622236_1280.jpg"),
    Path("static/uploads/6366a64e97724e86ae66a3df7128b20d_ai-generated-8700575_1280.jpg"),
    Path("static/uploads/1bfd9ee9a4924213bfe179bbcf7a05ef_ai-generated-8622236_1280.jpg"),
    Path("static/uploads/7955896945cf4f2faf6dd405ae82b5c1_ai-generated-8622236_1280.jpg"),
    Path("static/uploads/57f44228bdde428a964427d2a8ae7364_myphoto.edit2.png"),
    Path("static/uploads/f859aa7dc0244d87a86e792aca5ac5bc_myphoto.edit2.png"),
    Path("static/uploads/3b00c2b71824432f9e40e94d14811dee_shadow.jpeg"),
    Path("static/uploads/648499c13c274dbb95eb351c92c45eb8_shadow.jpeg"),
    Path("static/uploads/bc7fa74ab636409e831b6576ccf60999_shadow.jpeg"),
]

def hash_file(p):
    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

seen_hashes = set()
curated_records = []

def process_candidates(candidates, target_dir, label):
    count = 0
    for p in candidates:
        if not p.exists():
            continue
        try:
            with Image.open(p) as img:
                w, h = img.size
                # Verify minimum resolution >= 960px on smallest side or >= 1024px
                if max(w, h) < 1000:
                    print(f"Skipping {p.name}: resolution {w}x{h} below 1000px threshold.")
                    continue
                
                f_hash = hash_file(p)
                if f_hash in seen_hashes:
                    print(f"Skipping duplicate: {p.name} (SHA256 already present)")
                    continue
                
                seen_hashes.add(f_hash)
                dest = target_dir / f"{label}_{count:04d}{p.suffix.lower()}"
                shutil.copy2(p, dest)
                count += 1
                curated_records.append({
                    "filename": dest.name,
                    "original_source": p.name,
                    "label": label,
                    "width": w,
                    "height": h,
                    "sha256": f_hash,
                    "filesize": dest.stat().st_size,
                })
                print(f"Added [{label.upper()}]: {dest.name} ({w}x{h}, {dest.stat().st_size:,} bytes) from {p.name}")
        except Exception as e:
            print(f"Error processing {p.name}: {e}")
    return count

real_count = process_candidates(candidate_reals, real_dir, "real")
fake_count = process_candidates(candidate_fakes, fake_dir, "fake")

manifest_path = base_dir / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(curated_records, f, indent=4)

print("\n" + "=" * 80)
print(f"HIGH-RES BENCHMARK DATASET CURATION SUMMARY:")
print(f"Directory:           {base_dir.resolve()}")
print(f"Genuine Real Images: {real_count}")
print(f"Synthetic AI Images: {fake_count}")
print(f"Total Unique Images: {real_count + fake_count}")
print(f"Manifest written to: {manifest_path}")
print("=" * 80)
