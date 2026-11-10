import config
from pathlib import Path
from predict import predict_file

print("=" * 80)
print("EVALUATING GENERALIZATION ACROSS DISTINCT MEDIA CATEGORIES")
print("=" * 80)

categories = {
    "1. Genuine Camera Photographs": [
        "static/uploads/cf36c9647ce14667b790c93c3595fca9_IMG_20240207_174754.jpg",
        "static/uploads/4d3090911ad442b0ada6c6e82ab9e30c_bicky.jpg",
        "static/uploads/8c4b44dafb2743f78085ba70015ac7c3_1000032869.jpg",
        "static/uploads/06395a0d1bff4598ba04a89f4cb50b53_WhatsApp_Image_2026-10-22_at_9.42.38_AM_2.jpeg",
        "static/uploads/096a41120e054a1faee9cdef7a0ea8aa_Screenshot_20240622_181714.jpg",
    ],
    "2. CIFAKE Synthetic Benchmark (Stable Diffusion v1.4, 32x32)": [
        str(p) for p in list(Path("dataset/benchmark_cifake/test/fake").glob("*.png"))[:5]
    ],
    "3. High-Resolution Diffusion Generations (Pixabay/Midjourney/SDXL)": [
        "../ai-generated-8622236_1280.jpg",
        "static/uploads/6366a64e97724e86ae66a3df7128b20d_ai-generated-8700575_1280.jpg",
        "static/uploads/1bfd9ee9a4924213bfe179bbcf7a05ef_ai-generated-8622236_1280.jpg",
    ],
    "4. Web / Edited / Manipulated Images": [
        "../OIP.jpg",
        "static/uploads/35d8416e223e4847b777262f87fef6aa_OIP.jpg",
        "static/uploads/3b00c2b71824432f9e40e94d14811dee_shadow.jpeg",
        "static/uploads/57f44228bdde428a964427d2a8ae7364_myphoto.edit2.png",
    ]
}

for cat_name, file_list in categories.items():
    print(f"\n--- {cat_name} ---")
    correct_count = 0
    total_count = 0
    
    expected_real = "Genuine" in cat_name
    
    for f_str in file_list:
        p = Path(f_str)
        if not p.exists():
            continue
        total_count += 1
        res = predict_file(p, modality="image")
        pred = res["prediction"]
        prob = res["probability_real"]
        conf = res["confidence"] * 100
        
        is_correct = (pred == "Real") if expected_real else (pred == "Fake")
        if is_correct:
            correct_count += 1
        status = "CORRECT" if is_correct else "MISCLASSIFIED"
        print(f"  {p.name[:45]:<45} | Pred: {pred:<5} (Expected: {'Real' if expected_real else 'Fake'}) | Prob(Real): {prob:.4f} | Conf: {conf:5.1f}% | {status}")
        
    if total_count > 0:
        print(f"  Category Accuracy: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)")
