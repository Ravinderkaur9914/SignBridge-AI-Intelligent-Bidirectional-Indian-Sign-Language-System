"""
modules/gesture_recognizer.py
------------------------------
Loads a trained gesture recognition model and predicts
sign language gestures from hand landmark arrays.

Supports:
  - Keras model  (model/gesture_model.h5  + label_encoder.npy)  ← default
  - TFLite model (model/model.tflite      + label_encoder.npy)

UPDATED improvements:
  - Wrist-relative normalization in predict() matches training pipeline
  - Robust error handling

Train your model with train_model.py first.
"""

import os
import numpy as np

MODEL_DIR    = "model"
MODEL_KERAS  = os.path.join(MODEL_DIR, "gesture_model.h5")
MODEL_TFLITE = os.path.join(MODEL_DIR, "model.tflite")
LABELS_FILE  = os.path.join(MODEL_DIR, "label_encoder.npy")
NORM_MEAN    = os.path.join(MODEL_DIR, "norm_mean.npy")
NORM_STD     = os.path.join(MODEL_DIR, "norm_std.npy")


class GestureRecognizer:
    """
    Predicts a gesture label and confidence from a 42-float landmark vector.

    Pipeline (must match train_model.py exactly):
      raw landmarks (42,)
        → wrist-relative normalization   ← NEW: position & scale invariant
        → mean/std normalization          ← loaded from norm_mean/std.npy
        → model inference
        → label + confidence
    """

    def __init__(self):
        self.model      = None
        self.labels     = []
        self.model_type = None
        self.mean       = None
        self.std        = None
        self._load()

    # ── Load ──────────────────────────────────────────────────────────────────
    def _load(self):
        # Labels
        if not os.path.exists(LABELS_FILE):
            raise FileNotFoundError(
                f"Labels file not found: {LABELS_FILE}\n"
                "Run `python train_model.py` to train the model first."
            )
        self.labels = np.load(LABELS_FILE, allow_pickle=True).tolist()

        # Normalization stats
        if os.path.exists(NORM_MEAN) and os.path.exists(NORM_STD):
            self.mean = np.load(NORM_MEAN)
            self.std  = np.load(NORM_STD)

        # Keras model
        if os.path.exists(MODEL_KERAS):
            try:
                from tensorflow import keras
                self.model      = keras.models.load_model(MODEL_KERAS)
                self.model_type = "keras"
                print(f"[GestureRecognizer] Loaded Keras model — {len(self.labels)} classes")
                return
            except Exception as e:
                raise RuntimeError(f"Failed to load Keras model: {e}")

        # TFLite model
        if os.path.exists(MODEL_TFLITE):
            try:
                try:
                    import tflite_runtime.interpreter as tflite  # type: ignore
                except ImportError:
                    import tensorflow as tf
                    tflite = tf.lite
                self.model = tflite.Interpreter(model_path=MODEL_TFLITE)
                self.model.allocate_tensors()
                self.model_type = "tflite"
                print(f"[GestureRecognizer] Loaded TFLite model — {len(self.labels)} classes")
                return
            except Exception as e:
                raise RuntimeError(f"Failed to load TFLite model: {e}")

        raise FileNotFoundError(
            f"No model file found in '{MODEL_DIR}/'.\n"
            "Expected: gesture_model.h5  or  model.tflite\n"
            "Run `python train_model.py` first."
        )

    # ── Wrist normalization (MUST match train_model.py) ───────────────────────
    def _normalize_wrist(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Make landmarks position & scale independent.
        - Subtract wrist (keypoint 0) → translation invariant
        - Divide by max abs value     → scale invariant, range [-1, 1]

        Input/output: shape (42,)
        """
        pts   = landmarks.reshape(21, 2)
        wrist = pts[0].copy()
        pts   = pts - wrist                             # shift to wrist origin
        scale = np.max(np.abs(pts)) + 1e-8
        pts   = pts / scale                             # scale to [-1, 1]
        return pts.flatten()

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict(self, landmarks: np.ndarray) -> tuple:
        """
        Parameters
        ----------
        landmarks : np.ndarray  shape (42,) — flattened x,y for 21 keypoints

        Returns
        -------
        label      : str   — predicted gesture name
        confidence : float — prediction confidence [0.0 – 1.0]
        """
        if self.model is None or landmarks is None:
            return None, 0.0

        try:
            # Step 1 — Wrist-relative normalization (NEW)
            landmarks = self._normalize_wrist(landmarks)

            # Step 2 — Mean/std normalization
            if self.mean is not None and self.std is not None:
                landmarks = (landmarks - self.mean) / self.std

            X = landmarks.reshape(1, -1).astype(np.float32)

            # Step 3 — Inference
            if self.model_type == "keras":
                proba      = self.model.predict(X, verbose=0)[0]
                label_idx  = int(np.argmax(proba))
                confidence = float(proba[label_idx])

            elif self.model_type == "tflite":
                inp = self.model.get_input_details()
                out = self.model.get_output_details()
                self.model.set_tensor(inp[0]["index"], X)
                self.model.invoke()
                proba      = self.model.get_tensor(out[0]["index"])[0]
                label_idx  = int(np.argmax(proba))
                confidence = float(proba[label_idx])

            else:
                return None, 0.0

            label = self.labels[label_idx]
            return label, confidence

        except Exception as e:
            print(f"[GestureRecognizer] Predict error: {e}")
            return None, 0.0

    def get_labels(self) -> list:
        return self.labels