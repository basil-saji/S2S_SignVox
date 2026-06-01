import pandas as pd
import numpy as np
import tensorflow as tf
from .config import MAX_LEN, POINT_LANDMARKS, NUM_CLASSES, PAD, CHANNELS

def interp1d_(x, target_len, method='random'):
    length = tf.shape(x)[1]
    target_len = tf.maximum(1, target_len)
    if method == 'random':
        if tf.random.uniform(()) < 0.33:
            x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bilinear')
        else:
            if tf.random.uniform(()) < 0.5:
                x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'bicubic')
            else:
                x = tf.image.resize(x, (target_len, tf.shape(x)[1]), 'nearest')
    else:
        x = tf.image.resize(x, (target_len, tf.shape(x)[1]), method)
    return x

def tf_nan_mean(x, axis=0, keepdims=False):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

def tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_len=MAX_LEN, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    def call(self, inputs):
        if tf.rank(inputs) == 3:
            x = inputs[None, ...]
        else:
            x = inputs

        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1, 2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2)
        std = tf_nan_std(x, center=mean, axis=[1, 2], keepdims=True)

        x = (x - mean) / std

        if self.max_len is not None:
            x = x[:, :self.max_len]
        length = tf.shape(x)[1]
        x = x[..., :2]

        dx = tf.cond(tf.shape(x)[1] > 1, lambda: tf.pad(x[:, 1:] - x[:, :-1], [[0, 0], [0, 1], [0, 0], [0, 0]]), lambda: tf.zeros_like(x))
        dx2 = tf.cond(tf.shape(x)[1] > 2, lambda: tf.pad(x[:, 2:] - x[:, :-2], [[0, 0], [0, 2], [0, 0], [0, 0]]), lambda: tf.zeros_like(x))

        x = tf.concat([
            tf.reshape(x, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx, (-1, length, 2 * len(self.point_landmarks))),
            tf.reshape(dx2, (-1, length, 2 * len(self.point_landmarks))),
        ], axis=-1)

        x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
        return x

PREPROCESS = Preprocess()

def load_parquet_file(path):
    df = pd.read_parquet(path)
    frames = sorted(df["frame"].unique())

    f_idx = {f: i for i, f in enumerate(frames)}
    landmarks = np.full((len(frames), 543, 3), np.nan, dtype=np.float32)

    offsets = {
        "face": 0,
        "left_hand": 468,
        "pose": 489,
        "right_hand": 522
    }

    for row in df.itertuples():
        if row.type not in offsets: continue
        if row.type == "face" and row.landmark_index > 467: continue

        idx = offsets[row.type] + row.landmark_index
        f = f_idx[row.frame]

        landmarks[f, idx] = [row.x, row.y, row.z]

    return landmarks

def process_sample(path, label):
    landmarks = load_parquet_file(path)
    tensor = tf.convert_to_tensor(landmarks, dtype=tf.float32)

    x = PREPROCESS(tensor)[0].numpy()
    seq_len = x.shape[0]

    if seq_len > MAX_LEN:
        x = x[:MAX_LEN]
    elif seq_len < MAX_LEN:
        pad_len = MAX_LEN - seq_len
        x = np.pad(x, ((0, pad_len), (0, 0)), constant_values=PAD)

    y = np.zeros(NUM_CLASSES, dtype=np.float32)
    y[label] = 1.0

    return x.astype(np.float32), y

def create_datasets(train_df, val_df):
    X_train = np.zeros((len(train_df), MAX_LEN, CHANNELS), dtype=np.float32)
    y_train = np.zeros((len(train_df), NUM_CLASSES), dtype=np.float32)

    for i, (_, row) in enumerate(train_df.iterrows()):
        x, y = process_sample(row["path"], row["label"])
        X_train[i] = x
        y_train[i] = y

    X_val = np.zeros((len(val_df), MAX_LEN, CHANNELS), dtype=np.float32)
    y_val = np.zeros((len(val_df), NUM_CLASSES), dtype=np.float32)

    for i, (_, row) in enumerate(val_df.iterrows()):
        x, y = process_sample(row["path"], row["label"])
        X_val[i] = x
        y_val[i] = y
        
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.shuffle(2048).batch(32).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds
