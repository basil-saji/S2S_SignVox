# Dataset

## Overview

SignVox was trained and evaluated using the AI4Bharat INCLUDE dataset and a MediaPipe-Holistic landmark version derived from the dataset.

The objective of this phase was to develop an Indian Sign Language (ISL) recognition backbone using landmark-based deep learning techniques.

---

## Original Dataset

Dataset Name:
INCLUDE (INdian Corpus for Language Understanding through Examples)

Authors:
AI4Bharat

Repository:
https://github.com/AI4Bharat/INCLUDE

Language:
Indian Sign Language (ISL)

Task:
Isolated Sign Language Recognition

Number of Classes:
263

---

## Landmark Dataset Used

Training was performed on pre-extracted MediaPipe Holistic landmarks.

Dataset:
https://www.kaggle.com/datasets/swaptr/indian-sign-language-mediapipe-holistic-landmarks

This dataset contains:

- Face landmarks
- Pose landmarks
- Left hand landmarks
- Right hand landmarks

stored as parquet files for efficient training.

---

## Landmark Representation

Each sample is reconstructed into a tensor of shape:

(Frames, 543, 3)

where:

- 543 = total MediaPipe Holistic landmarks
- 3 = x, y, z coordinates

After preprocessing, samples are transformed into:

(384, 708)

which becomes the model input.

---

## Dataset Statistics

| Property | Value |
|-----------|--------|
| Classes | 263 |
| Total Samples | 4284 |
| Train Samples | 3427 |
| Validation Samples | 857 |
| Input Shape | (384, 708) |

---

## Notes

This repository does not redistribute the dataset.

Please obtain the dataset from the official sources linked above.
