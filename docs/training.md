# Training Pipeline

## Overview

SignVox uses a transfer learning approach to adapt a high-performing sign language recognition backbone to Indian Sign Language.

---

## Pipeline
```
Video
↓
MediaPipe Holistic
↓
Landmark Reconstruction
↓
Preprocessing
↓
CNN Feature Extractor
↓
Transformer Encoder
↓
Classification Head
↓
263 ISL Classes
```
---

## Input Representation

Raw Landmark Shape:

(Frames, 543, 3)

Preprocessed Shape:

(384, 708)

---

## Training Configuration

| Parameter | Value |
|------------|--------|
| Framework | TensorFlow 2.11 |
| Batch Size | 32 |
| Input Shape | (384, 708) |
| Classes | 263 |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Transfer Learning | Yes |

---

## Model Size

Total Parameters:

1,841,574

Approximate Size:

7.2 MB

---

## Training Strategy

The model was trained in multiple stages:

1. Initial sanity-check training
2. Backbone adaptation
3. Transfer learning on INCLUDE
4. Fine-tuning with checkpointing
5. Learning-rate reduction scheduling
6. Early stopping

Callbacks used:

- ModelCheckpoint
- EarlyStopping
- ReduceLROnPlateau

---

## Landmark Reconstruction Fix

During development a critical issue was discovered in the landmark reconstruction pipeline.

The dataset stores landmark indices locally per landmark type:

- Face
- Pose
- Left Hand
- Right Hand

The backbone architecture expects a global landmark layout.

The reconstruction pipeline was updated to correctly map landmark groups into their global positions before training.

This significantly improved inference behaviour and final model performance.

---

## Hardware

Training Environment:

- TensorFlow 2.11
- Python 3.10
- NVIDIA T4 GPUs
- Kaggle Notebooks

---

## Final Metrics

| Metric | Score |
|----------|----------|
| Accuracy | 95.80% |
| Top-5 Accuracy | 99.18% |
| Macro Precision | 96.68% |
| Macro Recall | 96.06% |
| Macro F1 | 95.89% |
