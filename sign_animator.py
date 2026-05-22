"""
modules/sign_animator.py
------------------------
Creates animated sign language display.
- Loops through sign images one by one with smooth transitions
- Returns frames as numpy arrays for Streamlit display
- Works with your existing dataset/signs folders

No extra libraries needed beyond opencv + numpy.
"""

import os
import cv2
import numpy as np
import time


# ── Folders where sign images live ────────────────────────────────────────────
SIGNS_ALPHA_DIR = "signs/alphabets"
SIGNS_WORDS_DIR = "signs/words"
DATASET_DIR     = "dataset"


class SignAnimator:
    """
    Loads sign images and animates them frame-by-frame.
    Use display_next_frame() inside a Streamlit loop.
    """

    def __init__(self, frame_size: tuple = (300, 300), fps: float = 1.5):
        self.frame_size   = frame_size    # (width, height)
        self.frame_delay  = 1.0 / fps     # seconds per sign
        self.signs        = []            # list of (label, image_path)
        self.current_idx  = 0
        self.last_time    = 0.0
        self.is_playing   = False

    # ── Load signs for a sentence ─────────────────────────────────────────────
    def load_sentence(self, sentence: str):
        """
        Break sentence into words/letters and find their sign images.
        sentence: e.g. "Hello I Need Help"
        """
        self.signs       = []
        self.current_idx = 0
        self.last_time   = 0.0

        words = sentence.strip().split()
        for word in words:
            path = self._find_image(word)
            if path:
                self.signs.append((word, path))
            else:
                # Try letter by letter if whole word not found
                for ch in word.upper():
                    lpath = self._find_image(ch)
                    if lpath:
                        self.signs.append((ch, lpath))
                    else:
                        self.signs.append((ch, None))  # placeholder

        self.is_playing = len(self.signs) > 0
        return self.signs

    # ── Get current animation frame ───────────────────────────────────────────
    def get_current_frame(self) -> tuple:
        """
        Returns (label, frame_image) for the current sign.
        Advances to next sign based on fps timing.
        Call this in a loop to animate.
        """
        if not self.signs:
            return ("", self._blank_frame("No Signs Loaded"))

        now = time.time()
        if now - self.last_time >= self.frame_delay:
            self.current_idx = (self.current_idx + 1) % len(self.signs)
            self.last_time   = now

        label, path = self.signs[self.current_idx]
        frame = self._load_sign_frame(label, path)
        return (label, frame)

    # ── Get all frames as a list (for preview) ────────────────────────────────
    def get_all_frames(self) -> list:
        """Returns list of (label, frame) for all signs in sentence."""
        frames = []
        for label, path in self.signs:
            frame = self._load_sign_frame(label, path)
            frames.append((label, frame))
        return frames

    # ── Build a grid image of all signs ───────────────────────────────────────
    def build_grid(self, cols: int = 5) -> np.ndarray:
        """
        Returns a single image showing all signs in a grid.
        Used for static preview.
        FIX: All frames are forced to identical (w, h+36) before stacking
             so np.hstack / np.vstack never sees mismatched dimensions.
        """
        if not self.signs:
            return self._blank_frame("No signs to display")

        w, h     = self.frame_size
        cell_h   = h + 36          # frame height + label bar height

        frames = []
        for label, path in self.signs:
            frame   = self._load_sign_frame(label, path)   # always (h, w, 3)
            labeled = self._add_label(frame, label)        # always (h+36, w, 3)

            # ── SAFETY: force exact cell size ─────────────────────────────────
            if labeled.shape[0] != cell_h or labeled.shape[1] != w:
                labeled = cv2.resize(labeled, (w, cell_h))

            frames.append(labeled)

        # Pad to fill last row with blank cells
        while len(frames) % cols != 0:
            blank = self._blank_frame("")                  # (h, w, 3)
            blank_labeled = self._add_label(blank, "")    # (h+36, w, 3)
            if blank_labeled.shape[0] != cell_h or blank_labeled.shape[1] != w:
                blank_labeled = cv2.resize(blank_labeled, (w, cell_h))
            frames.append(blank_labeled)

        # Stack into grid
        rows = []
        for i in range(0, len(frames), cols):
            row = np.hstack(frames[i:i + cols])
            rows.append(row)

        return np.vstack(rows)

    # ── Progress info ─────────────────────────────────────────────────────────
    def get_progress(self) -> tuple:
        """Returns (current_index, total_signs)."""
        return (self.current_idx + 1, len(self.signs))

    def reset(self):
        self.current_idx = 0
        self.last_time   = 0.0

    def stop(self):
        self.is_playing = False

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _find_image(self, name: str) -> str | None:
        """Search for sign image in all known folders."""
        exts    = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"]
        folders = [SIGNS_ALPHA_DIR, SIGNS_WORDS_DIR]

        # Also search in dataset folder (uses saved gesture images)
        dataset_gesture = os.path.join(DATASET_DIR, name)
        if os.path.isdir(dataset_gesture):
            imgs = [f for f in os.listdir(dataset_gesture)
                    if f.endswith((".jpg", ".jpeg", ".png"))]
            if imgs:
                return os.path.join(dataset_gesture, imgs[0])

        for folder in folders:
            for ext in exts:
                for variant in [name, name.lower(), name.upper(), name.title()]:
                    path = os.path.join(folder, variant + ext)
                    if os.path.exists(path):
                        return path
        return None

    def _load_sign_frame(self, label: str, path: str | None) -> np.ndarray:
        """Load and resize a sign image to exact frame_size, or return placeholder."""
        w, h = self.frame_size
        if path and os.path.exists(path):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.resize(img, (w, h))          # ← always exact (w, h)
                cv2.rectangle(img, (0, 0), (w-1, h-1), (100, 100, 200), 3)
                return img
        return self._placeholder_frame(label)

    def _placeholder_frame(self, label: str) -> np.ndarray:
        """Gray placeholder with label text when no image found."""
        w, h  = self.frame_size
        frame = np.ones((h, w, 3), dtype=np.uint8) * 45
        cv2.rectangle(frame, (10, 10), (w-10, h-10), (80, 80, 120), 2)

        text       = label[:8] if label else "?"
        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.5 if len(text) == 1 else 1.0
        thickness  = 4   if len(text) == 1 else 2
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
        x = (w - tw) // 2
        y = (h + th) // 2
        cv2.putText(frame, text, (x, y), font, font_scale,
                    (160, 160, 220), thickness)
        cv2.putText(frame, "No image", (w//2 - 35, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 100), 1)
        return frame

    def _blank_frame(self, msg: str) -> np.ndarray:
        """Black blank frame at exact frame_size."""
        w, h  = self.frame_size
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        if msg:
            cv2.putText(frame, msg, (20, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
        return frame

    def _add_label(self, frame: np.ndarray, label: str) -> np.ndarray:
        """Add a fixed 36px label bar below the frame."""
        w   = self.frame_size[0]           # always use configured width
        bar = np.ones((36, w, 3), dtype=np.uint8) * 30
        if label:
            (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            x = max(4, (w - tw) // 2)
            cv2.putText(bar, label, (x, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)
        return np.vstack([frame, bar])