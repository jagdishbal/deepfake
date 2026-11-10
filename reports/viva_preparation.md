# Deepfake Analysis: B.Tech Viva Preparation

**Degree:** B.Tech Computer Science Engineering (AI & ML)  
**Institution:** GIET University  
**Project:** Multimodal Deepfake Analysis  
**Purpose:** Technical defense guide based on the current repository implementation

## 1. Architecture Justification

**Q: How is the detection system organized?**

The repository contains modality-specific binary classifiers for images, videos, and audio. The image path uses a MobileNetV2 feature extractor; the video path applies a frame-level CNN and then an LSTM; the audio path classifies log-mel spectrograms with a compact 2D CNN. Each branch produces a fake/real score. This is a multimodal project in the sense that it handles three media modalities, but the current `models/multimodal_model.py` is a placeholder: the code does not yet implement a trained joint-fusion network. I would describe it as three unimodal pipelines, not claim cross-modal fusion.

**Q: Why was MobileNetV2 selected for image and video-frame feature extraction?**

The target machine has an NVIDIA RTX 3050 with 6 GB VRAM, so the architecture needs to limit convolutional computation and activation memory while retaining useful visual features. MobileNetV2 uses depthwise separable convolutions, inverted residual blocks, and linear bottlenecks to reduce computation relative to many conventional convolutional backbones. Starting from ImageNet weights also provides reusable low- and mid-level visual features, reducing the amount of task-specific data and training required.

The image input is fixed at 224 x 224 RGB. In the implemented transfer-learning setup, the MobileNetV2 backbone is frozen initially, followed by global average pooling, dropout 0.2, and a one-unit sigmoid classifier. Freezing the backbone avoids storing and updating its gradients during the initial training stage; the classifier head is trained with Adam at a learning rate of 1e-4 and binary cross-entropy. This is a practical accuracy/compute trade-off for a 6 GB GPU, not a guarantee that every batch size or video sequence will fit. If memory pressure occurs, reduce batch size or frame count before increasing hardware requirements.

**Q: What exactly is the video model, and how does it represent time?**

`models/video_model.py` applies a frozen MobileNetV2 through `TimeDistributed` to 10 frames of size 224 x 224. Global average pooling produces one feature vector per frame. An LSTM with 64 units and dropout 0.2 then processes the ordered feature sequence, followed by a sigmoid output. The CNN extracts spatial facial cues; the LSTM can model changes across sampled frames. Uniform sampling gives temporal coverage without decoding every frame.

**Q: Why not use a much larger model or train the visual backbone from scratch?**

A larger backbone would consume more compute and activation memory, and training from scratch would require substantially more labeled data and optimization time. Transfer learning starts from general visual representations and limits trainable parameters during the initial stage. MobileNetV2 is therefore a defensible baseline for constrained hardware and a student-project dataset. It is not a claim that MobileNetV2 is universally more accurate than larger architectures; that must be established by controlled experiments on the same split.

**Q: How much memory can video input alone consume?**

For float32 input, one 224 x 224 RGB frame uses `224 x 224 x 3 x 4` bytes, about 0.574 MiB. Ten frames are about 5.74 MiB per video before intermediate activations, model workspaces, and framework buffers. A batch of four video sequences is about 23 MiB for raw input tensors alone. At higher resolutions or with more frames, activations can dominate memory, which is why uniform sampling and a small video batch are important. The evaluator caps video batches at four.

**Q: Does the current execution prove that the RTX 3050 was used?**

Not by itself. The training notebook prints TensorFlow's detected GPU list, which should be checked on the defense machine. The recent evaluator run emitted TensorFlow's warning that native Windows GPU support is unavailable for its TensorFlow version, so that run should be described as CPU execution unless a supported configuration such as WSL2 or a compatible DirectML setup is verified. The architecture is selected for the stated 6 GB target, but hardware utilization must be demonstrated separately.

## 2. Multimodal Data Handling

**Q: Why not load complete raw MP4 and WAV files into memory?**

Compressed media files must be decoded before model input can be formed. Decoding an entire long video into full-resolution frames or loading a corpus of waveforms at once can consume substantial RAM and GPU memory, increase startup time, and make training vulnerable to out-of-memory failures. The pipeline keeps source assets on disk and decodes only the data needed for the current sample or batch. This also avoids storing redundant decoded copies of large media in memory.

**Q: How are video frames selected and prepared?**

The video preprocessing code uses OpenCV to read frame count, selects 10 approximately evenly spaced indices with `numpy.linspace`, seeks to those positions, converts BGR frames to RGB, resizes them to 224 x 224, and scales pixel values to [0, 1]. It emits one fixed-shape sequence per video. Failed reads are represented by zero frames in the production helper, so data-quality checks remain important; padding is a robustness behavior, not evidence that a corrupt video is valid.

**Q: Why use Mel-spectrograms instead of raw audio samples?**

A Mel-spectrogram represents short-time spectral energy on a frequency scale that is closer to human auditory perception. It makes local time-frequency patterns available to a 2D CNN while compressing a waveform that may contain tens of thousands of samples per second. The implemented audio extractor resamples to 16 kHz, uses a 3-second analysis window, computes 128 Mel bands up to 8 kHz (the Nyquist frequency at 16 kHz), converts power to decibels, normalizes the result, and adds a channel dimension. The audio CNN uses convolutional layers with 32, 64, and 128 filters, global average pooling, dropout 0.3, and a sigmoid output.

**Q: How does the notebook demonstrate that an MP4 audio track is readable?**

The audio visualization cell selects an MP4 from `dataset/train/audio/real`, invokes the FFmpeg executable bundled with `imageio-ffmpeg`, maps its audio stream to mono 16 kHz PCM WAV on standard output, and passes the bytes through an in-memory buffer to librosa. It plots the decoded waveform beside the log-Mel spectrogram. No intermediate WAV is written. This demonstrates successful decoding for the selected sample, not for every MP4 in the dataset.

**Q: Is MP4 audio currently accepted by the production audio dataset generator?**

There is an important implementation distinction. The notebook's visualization explicitly extracts AAC audio from an MP4 with FFmpeg. In contrast, `preprocessing/clean_audio.py` currently filters directory entries to `.wav`, `.mp3`, `.flac`, `.ogg`, and `.m4a`; `.mp4` is not in that extension set. Therefore, do not claim that the production audio generator directly trains on MP4 files. To support that, extract audio to a supported format or extend the generator to decode MP4 audio explicitly and test it. Also note that the extractor catches load errors and substitutes zeros, so logs and validation should distinguish valid silence from a decode failure.

## 3. Infrastructure & Optimization

**Q: What is the manifest strategy, and why is it useful for a roughly 50 GB dataset?**

`reports/dataset_manifest.csv` stores lightweight metadata rows: source filepath, modality, label, validity, and split. `preprocessing/build_manifest.py` performs integrity checks and assigns valid entries to stratified train, validation, and test splits, targeting 70/15/15 with `random_state=42`. The manifest itself indexes the source assets rather than embedding their bytes, so inventory and split decisions do not require another full dataset copy. It also provides a reproducible record of why a file belongs to a modality, class, and split.

**Q: Is the entire pipeline literally zero-duplication?**

No. The manifest stage avoids copying media just to index it, but the current physical-split population script uses `shutil.copy2` and creates two mirrored directory conventions (`dataset/{split}/{modality}` and `dataset/{modality}/{split}`). That materializes extra copies and consumes disk. `prepare_dataset.py` attempts hard links and falls back to copies, but `populate_physical_splits.py` is explicitly copy-based. A precise defense is: “The manifest avoids duplication at the catalog/indexing layer; physical convenience splits currently duplicate data.” A true end-to-end zero-copy design would use manifest-backed loaders or a single physical layout, and could use hard links where filesystem constraints permit.

**Q: How is lazy loading implemented? Is this a PyTorch DataLoader?**

No PyTorch `DataLoader` is used. Images are loaded in batches with Keras `image_dataset_from_directory`, then normalized and prefetched with `tf.data`. Video and audio use `tf.data.Dataset.from_generator`: the generator decodes one source item and yields its frame sequence or Mel-spectrogram, after which the dataset batches, optionally shuffles, and prefetches. The video and audio shuffle buffers are bounded (50 and 100 respectively) rather than holding the complete corpus in memory. `prefetch(tf.data.AUTOTUNE)` overlaps input preparation and model work where supported.

**Q: Why is execution constrained to the D: drive?**

The raw corpus and generated split folders are large enough to threaten the operating-system drive if temporary decoded files, caches, checkpoints, or duplicate media are placed there. Project paths, notebook outputs, model checkpoints, reports, and runtime cache/temp locations should therefore be explicitly rooted on D:. This is an operational storage constraint, not a property of the neural-network architecture. Before training, verify free space on D: and confirm cache and temporary-directory settings; do not assume that changing the working directory alone redirects every library cache.

**Q: How is split reproducibility handled, and what leakage risk remains?**

The manifest uses a fixed random seed and stratifies by label when enough samples exist, which makes the row-level split repeatable and helps preserve class proportions. However, a random file-level split does not automatically guarantee separation by subject, source video, or near-duplicate content. If multiple frames or related clips from one source can appear across splits, evaluation may be optimistic. A stronger protocol groups by source identity/video before splitting and records that group key in the manifest.

## 4. Evaluation Metrics

**Q: Why are precision, recall, and the confusion matrix more informative than accuracy?**

Accuracy is the fraction of all predictions that are correct. It can conceal class-specific failure, especially when class counts differ or the costs of false alarms and missed fakes differ. Precision answers “when the model predicts this class, how often is it correct?” Recall answers “of the examples truly in this class, how many did it find?” The confusion matrix exposes the actual error counts and their direction. The F1-score is the harmonic mean of precision and recall, so a high value requires both to be strong. Macro averages weight classes equally; weighted averages weight them by support.

**Q: What did the current image evaluation report show?**

On 231 test images (129 fake and 102 real), the report recorded 63.64% accuracy and ROC-AUC 0.7202. At the 0.5 decision threshold, the per-class results were:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Fake | 0.63 | 0.87 | 0.73 | 129 |
| Real | 0.67 | 0.34 | 0.45 | 102 |
| Macro average | 0.65 | 0.61 | 0.59 | 231 |
| Weighted average | 0.65 | 0.64 | 0.61 | 231 |

This result shows why accuracy alone is insufficient: the model identifies fake examples with substantially higher recall than it identifies real examples. The real-class recall of 0.34 is a clear weakness to investigate, not a result to obscure. Follow-up work should inspect class-specific errors, verify labels and preprocessing, evaluate thresholds on a validation set, and consider class weighting or additional representative data.

**Q: How should I explain the confusion matrix?**

The class order is fake then real, with rows as true labels and columns as predictions:

|  | Predicted fake | Predicted real |
|---|---:|---:|
| True fake | 112 | 17 |
| True real | 67 | 35 |

Thus, 17 fake samples were called real, while 67 real samples were called fake. If “fake” is treated as the positive class, these correspond to 17 false negatives and 67 false positives. The matrix makes those two error types concrete; which one is more costly depends on the intended deployment and operating threshold.

**Q: What does ROC-AUC add, and what does it not tell us?**

ROC-AUC summarizes ranking performance across decision thresholds using the true-positive and false-positive rates. A value of 0.7202 indicates better-than-random ranking on this test set, but it does not select a deployment threshold, guarantee calibration, or replace class-specific precision/recall. Threshold selection should use validation data and the application's error costs; the test set should remain reserved for final reporting.

## 5. Known Limitations and Defense

**Q: Can I claim the project performs joint audio-video fusion?**

Not from the current code. The repository has separate image, video, and audio model builders, but `models/multimodal_model.py` is a placeholder. A joint system would need an implemented fusion method, such as calibrated score-level fusion or a learned representation-level fusion layer, plus training and evaluation that preserve sample correspondence across modalities. Present the current work as a multi-modality analysis pipeline with independent branches, and identify fusion as future work.

**Q: What would you improve next?**

First, make MP4 audio decoding consistent between the notebook and production loader. Second, avoid mirrored physical copies by training from manifest paths or using one split layout; if copies are required, account for their storage explicitly. Third, split by subject/source to reduce leakage risk. Fourth, analyze the low real-class recall and choose a threshold using validation data. Finally, verify GPU availability on the defense machine and record the software versions, device, batch size, and exact test split with the results.

## Implementation References

- [Image model](../models/image_model.py)
- [Video model and streaming loader](../preprocessing/clean_video.py)
- [Audio model](../models/audio_model.py)
- [Audio feature extraction and generator](../preprocessing/clean_audio.py)
- [Manifest creation](../preprocessing/build_manifest.py)
- [Physical split population](../preprocessing/populate_physical_splits.py)
- [Image evaluation report](image_evaluation_report.txt)
