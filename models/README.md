# Models

This directory contains trained SIGN2SOUND backbone weights.

## Released Backbone

Input Shape:
(384, 708)

Classes:
263

Framework:
TensorFlow / Keras

Parameters:
1,841,574

---

## Performance

| Metric | Score |
|----------|----------|
| Accuracy | 95.80% |
| Top-5 Accuracy | 99.18% |
| Macro Precision | 96.68% |
| Macro Recall | 96.06% |
| Macro F1 | 95.89% |

---

## Usage

Load the model:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "signvox_isl_model.h5"
)
```

Generate predictions on preprocessed landmark sequences of shape:

```python
(384, 708)
```

---

## Notes

The released model is intended as a Phase 1 Indian Sign Language backbone and serves as the foundation for future real-time SIGN2SOUND systems.
