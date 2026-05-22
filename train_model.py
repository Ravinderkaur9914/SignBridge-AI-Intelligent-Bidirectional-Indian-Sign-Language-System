"""
train_model.py
--------------
Step 2: Model Training  (UPDATED — improved accuracy)
Loads the landmark CSV, trains a Dense neural network with Keras,
evaluates it, and saves:
    model/gesture_model.h5    – trained Keras model
    model/label_encoder.npy   – class label array (index → gesture name)
    model/norm_mean.npy        – normalization mean
    model/norm_std.npy         – normalization std

Improvements over original:
  1. Wrist-relative landmark normalization (position + scale independent)
  2. Data augmentation (3x samples via landmark noise)
  3. Class imbalance handling (class weights)
  4. Leaner, better-generalizing model architecture

Usage:
    python train_model.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, Input)
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
from tensorflow.keras.utils import to_categorical
import seaborn as sns

# ── Config ─────────────────────────────────────────────────────────────────────
DATASET_PATH = "dataset/landmarks.csv"
MODEL_DIR    = "model"
MODEL_PATH   = os.path.join(MODEL_DIR, "gesture_model.h5")
LABELS_PATH  = os.path.join(MODEL_DIR, "label_encoder.npy")
PLOT_PATH    = os.path.join(MODEL_DIR, "training_history.png")
CM_PATH      = os.path.join(MODEL_DIR, "confusion_matrix.png")

BATCH_SIZE   = 32
EPOCHS       = 150        # EarlyStopping will cut this short
TEST_SIZE    = 0.2
RANDOM_SEED  = 42
AUG_FACTOR   = 3          # augmentation multiplier (3 = 4x data total)
NOISE_STD    = 0.01       # landmark noise level for augmentation


# ══════════════════════════════════════════════════════════════════════════════
#  1. WRIST-RELATIVE NORMALIZATION  ← KEY IMPROVEMENT
#     Makes predictions position & scale independent.
#     Same gesture works whether hand is near/far, left/right of frame.
# ══════════════════════════════════════════════════════════════════════════════

def normalize_wrist(X: np.ndarray) -> np.ndarray:
    """
    Normalize landmarks relative to wrist (keypoint 0).
    Steps:
      - Subtract wrist position  → translation invariant
      - Divide by max absolute value → scale invariant, range [-1, 1]

    Input  shape: (N, 42)   — 21 keypoints × (x, y)
    Output shape: (N, 42)   — same, but normalized
    """
    X = X.reshape(-1, 21, 2)                                  # (N, 21, 2)
    wrist = X[:, 0:1, :]                                       # (N, 1, 2)
    X = X - wrist                                              # shift origin to wrist
    scale = np.max(np.abs(X), axis=(1, 2), keepdims=True) + 1e-8
    X = X / scale                                              # scale to [-1, 1]
    return X.reshape(-1, 42)                                   # (N, 42)


# ══════════════════════════════════════════════════════════════════════════════
#  2. DATA AUGMENTATION
#     Adds small random noise to landmarks → model becomes robust to
#     slight hand tremors, detection jitter, different people's hands.
# ══════════════════════════════════════════════════════════════════════════════

def augment_landmarks(X: np.ndarray, y: np.ndarray,
                      factor: int = 3, noise_std: float = 0.01):
    """
    Create `factor` additional noisy copies of the dataset.
    factor=3 → dataset becomes 4× original size.
    """
    X_list, y_list = [X], [y]
    for _ in range(factor):
        noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
        X_list.append(np.clip(X + noise, -1.0, 1.0))
        y_list.append(y)
    X_aug = np.vstack(X_list)
    y_aug = np.vstack(y_list)

    # Shuffle augmented data
    idx = np.random.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]


# ══════════════════════════════════════════════════════════════════════════════
#  3. DATASET LOADER
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(path: str):
    print(f"[INFO] Loading dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"[INFO] Total samples  : {len(df)}")
    print(f"[INFO] Gesture classes: {df['label'].nunique()}")
    print(f"[INFO] Class distribution:\n{df['label'].value_counts()}\n")

    X = df.drop("label", axis=1).values.astype(np.float32)

    le = LabelEncoder()
    y_int = le.fit_transform(df["label"].values)
    y = to_categorical(y_int, num_classes=len(le.classes_))

    return X, y, le.classes_, y_int


# ══════════════════════════════════════════════════════════════════════════════
#  4. MEAN/STD NORMALIZATION  (applied AFTER wrist normalization)
# ══════════════════════════════════════════════════════════════════════════════

def normalize_features(X_train, X_test):
    """Zero-mean, unit-variance normalization fitted on training data only."""
    mean = X_train.mean(axis=0)
    std  = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


# ══════════════════════════════════════════════════════════════════════════════
#  5. MODEL ARCHITECTURE  (leaner = better generalization for 42 inputs)
# ══════════════════════════════════════════════════════════════════════════════

def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    """
    Leaner fully-connected network — avoids overfitting on 42-dim input.
    Architecture: 42 → 128 → 128 → 64 → num_classes

    Why leaner than original (256→256→128→64)?
    - 42 input features don't need 256 neurons
    - Smaller = less overfitting, faster training, same/better accuracy
    """
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.3),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),

        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ══════════════════════════════════════════════════════════════════════════════
#  6. PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def plot_history(history, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"],     label="Train Acc")
    axes[0].plot(history.history["val_accuracy"], label="Val Acc")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["loss"],     label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Val Loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Training curves saved to {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(max(8, len(class_names)), max(6, len(class_names) - 2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"[INFO] Confusion matrix saved to {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    X, y, class_names, y_int_raw = load_dataset(DATASET_PATH)
    num_classes = len(class_names)
    print(f"[INFO] Input dim   : {X.shape[1]}")
    print(f"[INFO] Num classes : {num_classes}\n")

    # ── 2. Wrist normalization FIRST ──────────────────────────────────────────
    print("[INFO] Applying wrist-relative normalization...")
    X = normalize_wrist(X)

    # ── 3. Train / test split (before augmentation to avoid data leakage) ─────
    X_train, X_test, y_train, y_test, _, y_int_test = train_test_split(
        X, y, y_int_raw,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_int_raw,
    )
    print(f"[INFO] Train samples (before aug): {len(X_train)}")
    print(f"[INFO] Test  samples             : {len(X_test)}\n")

    # ── 4. Augment ONLY training data ─────────────────────────────────────────
    print(f"[INFO] Augmenting training data (factor={AUG_FACTOR})...")
    X_train, y_train = augment_landmarks(X_train, y_train,
                                         factor=AUG_FACTOR,
                                         noise_std=NOISE_STD)
    print(f"[INFO] Train samples (after  aug): {len(X_train)}\n")

    # ── 5. Mean/std normalization ─────────────────────────────────────────────
    X_train, X_test, mean, std = normalize_features(X_train, X_test)
    np.save(os.path.join(MODEL_DIR, "norm_mean.npy"), mean)
    np.save(os.path.join(MODEL_DIR, "norm_std.npy"),  std)
    print("[INFO] Normalization stats saved.\n")

    # ── 6. Class weights (handles imbalanced datasets) ────────────────────────
    y_int_train = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_int_train),
        y=y_int_train,
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"[INFO] Class weights computed for {len(class_weight_dict)} classes.\n")

    # ── 7. Build model ────────────────────────────────────────────────────────
    model = build_model(X_train.shape[1], num_classes)
    model.summary()

    # ── 8. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=6, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]

    # ── 9. Train ──────────────────────────────────────────────────────────────
    print("\n[INFO] Training started...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,   # ← handles class imbalance
        callbacks=callbacks,
        verbose=1,
    )

    # ── 10. Evaluate ──────────────────────────────────────────────────────────
    print("\n[INFO] Evaluating on test set...")
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[RESULT] Test Accuracy : {acc * 100:.2f}%")
    print(f"[RESULT] Test Loss     : {loss:.4f}\n")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # ── 11. Save labels ───────────────────────────────────────────────────────
    np.save(LABELS_PATH, class_names)
    print(f"[INFO] Label encoder saved to {LABELS_PATH}")

    # ── 12. Plots ─────────────────────────────────────────────────────────────
    plot_history(history, PLOT_PATH)
    plot_confusion_matrix(y_true, y_pred, class_names, CM_PATH)

    print("\n✅ ALL DONE!")
    print(f"   Model  → {MODEL_PATH}")
    print(f"   Labels → {LABELS_PATH}")
    print(f"   Run 'streamlit run app.py' to launch.\n")


if __name__ == "__main__":
    main()