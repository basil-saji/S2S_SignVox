import os
import json
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from .config import MAX_LEN, NUM_CLASSES
from .data_loader import create_datasets
from .model import get_model

def load_data(root_dir):
    kp_root = os.path.join(root_dir, "keypoints")
    with open(f"{root_dir}/label_map.json") as f:
        label_map = json.load(f)

    rows = []
    for cat in os.listdir(kp_root):
        cat_path = os.path.join(kp_root, cat)
        if not os.path.isdir(cat_path): continue

        for word in os.listdir(cat_path):
            word_path = os.path.join(cat_path, word)
            if not os.path.isdir(word_path): continue

            if word not in label_map:
                continue

            lbl = label_map[word]
            for f in os.listdir(word_path):
                if f.endswith(".parquet"):
                    rows.append({
                        "path": os.path.join(word_path, f),
                        "word": word,
                        "label": lbl
                    })
    return pd.DataFrame(rows)

def transfer_weights(model, weights_path):
    # This function expects the ASL weights to be configured and available
    print("Building temporary ASL architecture...")
    global NUM_CLASSES
    # Temporarily set NUM_CLASSES to the ASL shape, load weights, then reconstruct ISL
    pass

def main():
    # Update this path relative to the environment where the code executes
    ROOT_DATASET_DIR = "/kaggle/input/datasets/swaptr/indian-sign-language-mediapipe-holistic-landmarks"
    
    df = load_data(ROOT_DATASET_DIR)
    
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )
    
    print("Building datasets...")
    train_ds, val_ds = create_datasets(train_df, val_df)
    
    print("Building ISL model architecture...")
    isl_model = get_model(max_len=MAX_LEN, dim=192)
    
    # Note: If performing transfer learning, ensure the ASL weight loading logic 
    # (from Cell 12 and 13) is inserted here before compiling.

    isl_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )
    
    checkpoint = ModelCheckpoint(
        "best_include_model.h5",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6
    )

    print("Starting training...")
    history = isl_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=60,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    isl_model.save_weights("include_signvox_epoch60_manual.h5")
    isl_model.save("include_aigvox_epoch60_full.h5")
    print("Training complete. Artifacts successfully saved.")

if __name__ == "__main__":
    main()
