# SignVox: Lightweight Indian Sign Language Recognition through Transfer Learning

![Status](https://img.shields.io/badge/Status-Active-success)
![Phase](https://img.shields.io/badge/Phase-1%20Backbone-blue)
![Dataset](https://img.shields.io/badge/Dataset-AI4Bharath_INCLUDE-orange)
![Framework](https://img.shields.io/badge/TensorFlow-2.x-FF6F00)
![Language](https://img.shields.io/badge/Python-3.10.x-blue)

---

## Overview

SignVox is an ongoing research and development project focused on building an accessible, lightweight, and real-time Indian Sign Language (ISL) recognition system.

The long-term goal is to develop a Sign-to-Speech communication platform capable of understanding Indian Sign Language and converting it into natural language and speech output.

This repository currently contains the **Phase 1 Backbone Development** work, where a state-of-the-art landmark-based sign language recognition architecture was adapted to Indian Sign Language through transfer learning.

---

## Motivation

Communication barriers remain a major challenge for members of the Deaf and Hard-of-Hearing community.

While substantial research exists for American Sign Language (ASL), comparatively fewer open-source solutions are available for Indian Sign Language (ISL).

SignVox aims to bridge this gap by:

- Leveraging modern deep learning techniques
- Building on proven sign-language recognition architectures
- Supporting real-time deployment on consumer hardware
- Creating an extensible foundation for future Sign-to-Speech systems

---

## Project Roadmap

### Phase 1 — ISL Recognition Backbone 

- Landmark-based sign recognition
- Transfer learning from a state-of-the-art sign language recognition backbone
- Fine-tuning on the INCLUDE dataset
- Evaluation and benchmarking

### Phase 2 — Real-Time Inference 

- Webcam-based recognition
- Live prediction pipeline
- Confidence estimation
- Performance optimization

### Phase 3 — Intent Recognition 

- Recognition of conversational intents
- Domain-specific communication assistance

### Phase 4 — Sign-to-Speech 

- Natural language generation
- Speech synthesis
- Assistive communication platform

---

## Dataset

### INCLUDE Dataset

This project uses the AI4Bharat INCLUDE dataset.

| Property | Value |
|-----------|--------|
| Language | Indian Sign Language |
| Classes | 263 |
| Samples | 4284 videos |
| Format | MediaPipe Holistic Landmarks |
| Type | Isolated Sign Recognition |

Dataset Repository:

https://github.com/AI4Bharat/INCLUDE

---

## Architecture

The backbone is based on a transfer-learning approach:

```text
Video
   ↓
MediaPipe Holistic
   ↓
Landmark Reconstruction
   ↓
Feature Preprocessing
   ↓
1D CNN Blocks
   ↓
Transformer Blocks
   ↓
Classification Head
   ↓
263 ISL Classes
```

### Model Characteristics

| Property | Value |
|----------|---------|
| Parameters | ~1.84 Million |
| Input Shape | (384, 708) |
| Output Classes | 263 |
| Framework | TensorFlow / Keras |
| Training Strategy | Transfer Learning |

---

## Transfer Learning Strategy

Rather than training from scratch, SignVox leverages knowledge from a high-performing sign-language recognition backbone and adapts it to Indian Sign Language.

```text
Pretrained Sign Language Backbone
            ↓
Replace Classification Head
            ↓
Fine-tune on INCLUDE Dataset
            ↓
Indian Sign Language Backbone
```

This significantly reduces training requirements while improving convergence and performance.

---

## Results

### Phase 1 Backbone Performance

| Metric | Score |
|----------|----------|
| Top-1 Accuracy | 95.80% |
| Top-5 Accuracy | 99.18% |
| Macro Precision | 96.68% |
| Macro Recall | 96.06% |
| Macro F1 Score | 95.89% |

---

## Key Engineering Challenge

During development, a major issue was discovered involving landmark reconstruction.

The original backbone architecture expects a global landmark topology, while the INCLUDE dataset stores landmark indices locally within individual landmark groups.

This mismatch resulted in incorrect landmark placement during tensor reconstruction.

After identifying and correcting the reconstruction pipeline, the backbone achieved significantly improved performance and stable inference behaviour.

This investigation highlighted the importance of preserving geometric consistency when adapting pretrained sign-language recognition architectures to new datasets.

---

## Repository Structure

```text
SignVox/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── notebooks/
│   └── SignVox_phase1_backbone.ipynb
│
├── reports/
│   ├── Phase1_Backbone_Report.pdf
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── f1_distribution.png
│   └── classification_report.txt
│
├── docs/
│   ├── dataset.md
│   ├── training.md
│   └── roadmap.md
│
├── assets/
│   ├── architecture.png
│   ├── demo.gif
│   └── SignVox_logo.png
│
├── data/
│   └── README.md
│
└── models/
    └── README.md
```

---

## Installation

Clone the repository:

```bash
git clone git@github.com:basil-saji/S2S_SignVox.git

cd S2S_SignVox
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducing Training

1. Obtain the INCLUDE dataset.
2. Place dataset files according to the notebook instructions.
3. Open:

```text
notebooks/SignVox_phase1_backbone.ipynb
```

4. Execute the notebook sequentially.

The notebook contains:

- Dataset preparation
- Landmark reconstruction
- Feature preprocessing
- Transfer learning pipeline
- Training
- Evaluation
- Visualization

---

## Future Work

- Real-time webcam inference
- Continuous sign recognition
- Intent recognition
- Sign-to-Speech generation
- Edge-device optimization
- Mobile deployment

---

## Contributors

### Team SignVox

- Basil Saji
- Contributors to be added

---

## Acknowledgements

This work builds upon:

- AI4Bharat INCLUDE Dataset
- MediaPipe Holistic
- TensorFlow
- Keras
- Open-source sign language recognition research

Special thanks to the researchers, developers, and open-source contributors whose work helped make this project possible.

---

## License

This project is released under the MIT License.

See the `LICENSE` file for details.

---

## Citation

If you use this work in research, educational projects, or derivative systems, please cite this repository and acknowledge the SignVox project.

---

## Vision

SignVox aims to evolve beyond isolated sign recognition into a complete assistive communication platform capable of:

- Understanding Indian Sign Language
- Recognizing conversational intent
- Translating signs into natural language
- Generating speech output in real time

Our goal is to contribute toward more accessible and inclusive communication technologies powered by AI.

---

**SignVox — Empowering Communication Through AI and Indian Sign Language.** 🚀
