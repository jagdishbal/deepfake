import re

with open('predict.py', 'r') as f:
    code = f.read()

# Add get_threshold function
threshold_func = """
import json

def get_threshold(modality: str) -> float:
    try:
        with open(Path(__file__).parent / 'models' / 'calibration.json', 'r') as f:
            data = json.load(f)
            return data.get(modality.lower(), {}).get('threshold', 0.5)
    except Exception:
        return 0.5
"""
code = code.replace("REAL_PROBABILITY_THRESHOLD = 0.5", threshold_func)

# Modify _classify_probability definition
code = code.replace("def _classify_probability(probability_real: float) -> str:", "def _classify_probability(probability_real: float, modality: str) -> str:")

# Modify _classify_probability body
code = code.replace('return "Real" if probability_real >= REAL_PROBABILITY_THRESHOLD else "Fake"', 'return "Real" if probability_real >= get_threshold(modality) else "Fake"')

# In predict_video, we need to pass 'video'
code = re.sub(r'def predict_video\(.*?\):.*?prediction = _classify_probability\(probability_real\)', lambda m: m.group(0).replace('_classify_probability(probability_real)', '_classify_probability(probability_real, "video")'), code, flags=re.DOTALL)

# In predict_file, we need to pass modality
code = re.sub(r'def predict_file\(.*?\):.*?prediction = _classify_probability\(probability_real\)', lambda m: m.group(0).replace('_classify_probability(probability_real)', '_classify_probability(probability_real, modality)'), code, flags=re.DOTALL)

with open('predict.py', 'w') as f:
    f.write(code)
