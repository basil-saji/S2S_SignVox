# Architecture

## Overview

SignVox employs a landmark-based deep learning pipeline for Indian Sign Language (ISL) recognition.

Instead of operating directly on raw video frames, the system converts signing videos into compact skeletal landmark representations using MediaPipe Holistic. These landmarks are then processed by a transfer-learned neural architecture combining convolutional and transformer-based components.

The resulting backbone is lightweight, efficient, and suitable for future real-time deployment.

---

## System Architecture

<p align="center">
  <img width="1836" height="2909" alt="signvox_isl_architecture" src="https://github.com/user-attachments/assets/0ea9916f-78ef-48e7-ace3-f1c1f987cc40" />
</p>

---

## Pipeline

### 1. Video Input

The pipeline begins with an RGB video containing an isolated Indian Sign Language gesture.

The video serves as the raw source from which body and hand motion information is extracted.

---

### 2. MediaPipe Holistic

MediaPipe Holistic is used to extract skeletal landmarks from each frame.

The extracted landmarks include:

- Face landmarks
- Pose landmarks
- Left-hand landmarks
- Right-hand landmarks

These landmarks provide a compact representation of the signer while discarding unnecessary visual information such as background details, lighting conditions, and clothing appearance.

---

### 3. Landmark Reconstruction

The landmark dataset stores indices locally for each landmark group.

To ensure compatibility with the pretrained backbone architecture, landmarks are reconstructed into a unified global topology consisting of 543 landmark positions.

The final structure follows:

| Component | Landmark Range |
|------------|----------------|
| Face | 0 – 467 |
| Left Hand | 468 – 488 |
| Pose | 489 – 521 |
| Right Hand | 522 – 542 |

This step preserves the spatial relationships expected by the backbone.

---

### 4. Preprocessing Pipeline

The reconstructed landmarks undergo preprocessing before being fed into the model.

Operations include:

- Landmark selection
- Missing value handling
- Normalization
- Temporal alignment
- Sequence padding/truncation
- Feature engineering

After preprocessing, each sample is transformed into a fixed-size representation:

```text
(384, 708)
```

This becomes the model input.

---

### 5. Transfer Learning Backbone

SignVox adopts a transfer learning strategy rather than training a model entirely from scratch.

A pretrained sign language recognition backbone serves as the starting point, allowing the network to leverage previously learned spatio-temporal representations.

The classification head is replaced and fine-tuned on the INCLUDE dataset.

---

### 6. CNN Feature Extractor

The convolutional component learns local spatial relationships between landmarks.

Its responsibilities include:

- Local feature extraction
- Motion pattern encoding
- Landmark interaction modelling

The CNN stage transforms raw landmark sequences into richer feature representations.

---

### 7. Transformer Encoder

The transformer component models long-range temporal dependencies across the signing sequence.

Responsibilities include:

- Temporal attention
- Sequence understanding
- Context modelling
- Motion dependency learning

This stage enables the model to distinguish signs that may share similar poses but differ in movement patterns.

---

### 8. Classification Head

The final feature representation is passed through fully connected layers to produce class probabilities.

Output:

```text
263 Indian Sign Language Classes
```

The class with the highest probability is selected as the prediction.

---

## Model Statistics

| Property | Value |
|-----------|---------|
| Parameters | 1.84 Million |
| Input Shape | (384, 708) |
| Output Classes | 263 |
| Framework | TensorFlow 2.11 |
| Dataset | AI4Bharat INCLUDE |

---

## Training Strategy

The backbone was trained using transfer learning on the AI4Bharat INCLUDE dataset.

Training employed:

- Adam Optimizer
- Categorical Crossentropy Loss
- ModelCheckpoint
- EarlyStopping
- ReduceLROnPlateau

The model was fine-tuned through multiple training stages until convergence.

---

## Performance

| Metric | Score |
|----------|----------|
| Accuracy | 95.80% |
| Top-5 Accuracy | 99.18% |
| Macro Precision | 96.68% |
| Macro Recall | 96.06% |
| Macro F1 Score | 95.89% |

---

## Future Extensions

The current architecture forms the foundation for future SignVox modules, including:

- Real-time webcam inference
- Continuous sign recognition
- Intent recognition
- Sign-to-Speech generation
- Edge-device deployment

The Phase 1 backbone establishes a strong foundation for building a complete Indian Sign Language communication platform.
