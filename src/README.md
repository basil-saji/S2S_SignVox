# Source Code Documentation

This directory contains the core modularized Python scripts for the SignVox project. These scripts were extracted and refactored from the primary exploratory Jupyter Notebook to support automated code review systems, CI/CD pipelines, and standard production deployments.

## Directory Structure and Module Descriptions

The codebase is divided into logical components to separate configuration, data processing, architectural definition, and execution logic.

* **`config.py`**
    Contains all global configuration variables, hyperparameters (e.g., learning rate, batch size, maximum sequence length), and standard constants for MediaPipe holistic landmarks. Modify this file to adjust training parameters without altering the core logic.

* **`data_loader.py`**
    Encapsulates the data ingestion and preprocessing pipeline. It includes the custom `Preprocess` TensorFlow Keras layer, functions for parsing `.parquet` files, and the logic to build the optimized `tf.data.Dataset` objects for training and validation.

* **`model.py`**
    Holds the neural network architecture definitions. It includes all custom layers such as the Efficient Channel Attention (`ECA`) module, `CausalDWConv1D`, `MultiHeadSelfAttention`, and the `TransformerBlock`. The primary `get_model()` function is defined here to instantiate the compiled architecture.

* **`train.py`**
    The main execution entry point. This script imports the necessary components from the other modules, loads the dataset paths, initiates the dataset creation, compiles the model, and runs the training loop with the defined callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau).

## Execution Instructions

To execute the training pipeline using these modular scripts, ensure you are in the root directory of the repository (one level up from `src/`) so that relative paths and module imports resolve correctly.

1.  Ensure all dependencies from the root `requirements.txt` are installed.
2.  Verify the dataset paths in `src/train.py` or `src/config.py` point to your local or environment data directories.
3.  Run the training module:

```bash
python -m src.train
