# Deepfake Model Audit and Evaluation

Evaluated on 2026-10-22 using the active inference code and the existing `dataset/<modality>/test` splits. Thresholds were selected from validation data only. No test examples were used to choose thresholds.

## Label and output semantics

- Image folders explicitly map `fake` to 0 and `real` to 1.
- Audio and video generators enumerate `fake` then `real`, also mapping fake to 0 and real to 1.
- All three classifiers end in a sigmoid unit trained with binary cross-entropy. Its scalar output is therefore interpreted as the probability of class 1 (REAL); the prediction code validates that it is in [0, 1].
- Displayed confidence is the probability assigned to the class selected by the modality's validation threshold: `probability_real` for REAL, otherwise `1 - probability_real`. Because thresholds need not be 0.50, this can differ from the larger class probability. These sigmoid scores have not been established as calibrated probabilities of correctness.

## Active architectures and preprocessing

| Modality | Architecture | Training and inference input |
|---|---|---|
| Image | Frozen MobileNetV2, global average pooling, dropout, sigmoid dense head | RGB, resize to 224x224, float pixels in [0, 1] |
| Video | TimeDistributed MobileNetV2, global average pooling, LSTM, sigmoid dense head | 10 uniformly sampled frames; BGR-to-RGB, resize to 224x224, float pixels in [0, 1] |
| Audio | 2D CNN, global average pooling, dense layer, sigmoid | Librosa mono decode/resample to 16 kHz; first 3 seconds padded/truncated; 128-bin mel power spectrogram, dB conversion, per-example min-max normalization to [0, 1] |

No face detector or face crop is used; the image model receives whole images and the video model receives full frames. Audio training and inference use the same feature function, duration, sample rate, and scaling. No second spectrogram normalization was found. The CNN outputs one sequence-level video score; current inference passes the whole frame sequence to the video model rather than averaging image-model frame scores.

One remaining image/video issue is the MobileNetV2 input range. The runtime `preprocess_input` maps pixel values from [0, 255] to [-1, 1], while this project consistently feeds [0, 1] in both training and inference. Thus train/inference agree with each other, but the inputs do not match the pretrained ImageNet backbone's expected range. Correcting this requires retraining both visual checkpoints; it was not changed in inference alone because that would make current weights incompatible.

## Findings by modality

### Image

The inference normalization mismatch (inference had used [-1, 1] while training used [0, 1]) was corrected to match the training input. Validation macro-F1 calibration selected 0.34 rather than the old 0.50 threshold. The current checkpoint was not retrained.

### Video

The video route was corrected to load the video checkpoint and submit the expected 10-frame sequence. The model output semantics and 0.54 validation macro-F1 threshold are consistent with the labels. Held-out ranking performance is limited, so a threshold change would not fix the underlying generalization issue.

The user's Hailuo cinematic animation was reproduced from the saved upload: the model returned `Real`, `probability_real=0.697537`, and predicted-class score `0.697537` on ten frames. Image-model per-frame scores ranged from 0.1581 to 0.5412, demonstrating that a single confidence figure hides substantial variation. The video model was trained on face-manipulation samples, not on fully AI-generated/animated-versus-camera-recorded videos. This sample is outside that validated domain; there is no evidence to claim the model can reliably judge whether it is AI-generated. The result page now warns about that limitation rather than treating “Real” as authentication. The output label was not forcibly changed.

### Audio

The audio pipeline has consistent training/inference preprocessing and correct fake=0, real=1 labels. The active checkpoint is poor: its held-out ROC-AUC is below 0.50 and it classifies 23 of 25 fake test samples as REAL at the calibrated 0.48 threshold. The historical training log contains only one epoch.

An isolated, class-weighted candidate was trained for 8 epochs with early stopping, without replacing the active checkpoint. It had validation AUC 0.450 and validation macro-F1 0.381, versus 0.581 AUC and 0.405 macro-F1 for the active checkpoint. On the same existing test split, the candidate's validation-selected 0.51 threshold gave 0.532 accuracy and 0.347 macro-F1, versus 0.468 accuracy and 0.377 macro-F1 for the active model. The candidate is not integrated; the original checkpoint remains active. Existing training histories are present for all modalities; image/video ran 15/10 epochs, while the original audio log contains one epoch. The trainer uses best-validation-loss checkpointing and early stopping with patience 3.

## Validation-selected thresholds

`calibrate_thresholds.py` now selects the threshold that maximizes macro-F1 (equal weight to both labels), with ties resolved toward 0.50. It uses validation splits only.

| Modality | Previous threshold | Selected threshold | Validation macro-F1 |
|---|---:|---:|---:|
| Image | 0.50 | 0.34 | 0.7364 |
| Video | 0.54 | 0.54 | 0.7771 |
| Audio | 0.48 | 0.48 | 0.4052 |

## Test-set results at the saved thresholds

Confusion matrices use rows = ground truth `[Fake, Real]`, columns = prediction `[Fake, Real]`.

| Modality | N | Accuracy | Macro precision | Macro recall | Macro F1 | ROC-AUC | Confusion matrix | FPR Fake→Real | FNR Real→Fake |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|
| Image | 231 | 0.7100 | 0.7251 | 0.6880 | 0.6874 | 0.8026 | `[[113, 16], [51, 51]]` | 0.1240 | 0.5000 |
| Video | 58 | 0.6034 | 0.6027 | 0.6024 | 0.6024 | 0.6202 | `[[19, 11], [12, 16]]` | 0.3667 | 0.4286 |
| Audio | 47 | 0.4681 | 0.4826 | 0.4945 | 0.3767 | 0.4491 | `[[2, 23], [2, 20]]` | 0.9200 | 0.0909 |

For the image checkpoint, changing only the threshold from 0.50 to the validation-selected 0.34 changed the same test set from 0.6017 accuracy / 0.4705 macro-F1 to 0.7100 accuracy / 0.6874 macro-F1. ROC-AUC remained 0.8026 because thresholding does not change score ranking. This is a diagnostic comparison, not evidence of improved model generalization.

All 115 misclassified test records are listed in [test_misclassifications.csv](./test_misclassifications.csv), including ground truth, predicted class, raw REAL probability, predicted-class confidence, threshold, and preprocessing pipeline.

## Dataset integrity and retraining decision

| Modality | Train (Fake / Real) | Validation (Fake / Real) | Test (Fake / Real) |
|---|---:|---:|---:|
| Image | 629 (389 / 240) | 153 (96 / 57) | 231 (129 / 102) |
| Video | 146 (80 / 66) | 36 (20 / 16) | 58 (30 / 28) |
| Audio | 114 (68 / 46) | 26 (16 / 10) | 47 (25 / 22) |

An exact duplicate image was found across the existing train and test splits. Many image filenames and a smaller number of audio/video stems also recur across splits; matching names alone do not prove identical content, but the exact duplicate confirms test leakage. The raw source tree contains 162,210 files, so it was not reprocessed or duplicated. Because split/group leakage and source-label mapping need to be cleaned before a trustworthy model comparison, image and video were not automatically retrained. The small audio candidate did not improve validation macro-F1, so it was rejected.

The dataset preparation code was corrected to use the nearest class directory, map FakeAVCeleb's `FakeVideo/RealVideo` and `FakeAudio/RealAudio` labels according to modality, preserve labels when importing a single class folder, and skip unlabeled files rather than silently assigning them REAL. Large future preparations now keep files from the same detected identity/clip group in one split. Existing prepared splits were not regenerated.

The active checkpoints remain:

- `models/saved_models/image/best_model.h5`
- `models/saved_models/video/best_model.h5`
- `models/saved_models/audio/best_model.h5`

The rejected audio candidate is retained separately at `runtime/candidate_models/audio/best_model.h5`.

## Validation and limitations

- The complete suite (`python -m pytest -q tests`) passed: 21 tests, including mocked image/audio/video inference, confidence boundaries, invalid outputs, zero-byte uploads, corrupt images, strict video decoding, and upload-to-result behavior.
- Python syntax parsing passed for all nine changed Python files.
- Flask homepage returned HTTP 200. The upload-to-result test verifies the exact saved record and displayed score using mocked inference and database access.
- Real inference succeeded for one test sample per modality using the active `best_model.h5` checkpoints. This is a smoke test, not an accuracy metric; the image and audio Fake examples were predicted Real.
- Held-out evaluation was run for all modalities; see the table above and the generated modality evaluation reports. TensorFlow 2.21 on this native Windows environment reports GPU support unavailable, so inference and evaluation ran on CPU.
- The project report documents an exact duplicate image across existing train/test splits. The prepared split results are therefore provisional and should be regenerated and rechecked before making model-quality claims. `reports/dataset_statistics.csv` is stale: it lists zero samples even though the prepared splits contain labeled test samples.
- No historical upload CSV with `Uploaded At`, `Filename`, `Media Type`, `Prediction`, and `Confidence (%)` was present in the project. `test_misclassifications.csv` is a labeled list of model errors, not the requested upload history.
