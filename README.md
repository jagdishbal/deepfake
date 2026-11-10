# Deepfake Analysis

Binary labels are consistent across the model pipeline: `fake = 0`, `real = 1`.
Image and video inputs use MobileNetV2 preprocessing (`[0, 255]` to `[-1, 1]`)
for both training and inference.

Run held-out evaluation with `python evaluate.py --modality image` (or `video`
or `audio`). Train experimental checkpoints under `runtime/candidate_models/`
by passing `--checkpoint-path` to `train.py`; compare candidates on validation
and test data before replacing an active checkpoint in `models/saved_models/`.
