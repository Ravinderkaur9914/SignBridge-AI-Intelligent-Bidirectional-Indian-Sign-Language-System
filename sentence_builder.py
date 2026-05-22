"""
modules/sentence_builder.py
----------------------------
Converts a stream of per-frame gesture predictions into stable words,
then assembles them into a sentence.

Logic:
  - A gesture must be held for `hold_frames` consecutive frames → confirmed word
  - After confirming, a `cooldown_frames` pause prevents the same
    gesture from re-triggering immediately
  - Duplicate consecutive words are suppressed
"""


class SentenceBuilder:
    """
    Stateful builder: call update() once per camera frame.
    """

    def __init__(self, hold_frames: int = 15, cooldown_frames: int = 30):
        self.hold_frames     = hold_frames
        self.cooldown_frames = cooldown_frames

        self._current_label  = None
        self._hold_counter   = 0
        self._cooldown_left  = 0
        self._last_word      = None
        self._sentence_ready = False

    # ── Per-frame update ──────────────────────────────────────────────────────
    def update(self, label: str | None) -> str | None:
        """
        Call with the predicted label (or None) for each frame.

        Returns
        -------
        new_word : str | None — the confirmed new word, or None if not yet ready
        """
        self._sentence_ready = False

        # Tick down cooldown
        if self._cooldown_left > 0:
            self._cooldown_left -= 1
            return None

        if label is None:
            self._current_label = None
            self._hold_counter  = 0
            return None

        # New gesture started
        if label != self._current_label:
            self._current_label = label
            self._hold_counter  = 1
            return None

        # Same gesture — increment hold counter
        self._hold_counter += 1

        if self._hold_counter >= self.hold_frames:
            # Suppress duplicate consecutive words
            if label == self._last_word:
                self._hold_counter  = 0
                self._cooldown_left = self.cooldown_frames
                return None

            # Confirmed word
            self._last_word     = label
            self._hold_counter  = 0
            self._cooldown_left = self.cooldown_frames

            # Treat "DONE" / "END" / "." as sentence end signal
            if label.upper() in ("DONE", "END", "."):
                self._sentence_ready = True

            return label

        return None

    # ── Sentence-end signal ───────────────────────────────────────────────────
    def is_sentence_ready(self) -> bool:
        """Returns True for one frame when a sentence-end gesture is detected."""
        return self._sentence_ready

    def clear(self):
        """Reset sentence-end flag after it has been consumed."""
        self._sentence_ready = False

    def reset(self):
        """Full reset — call when user clears the sentence."""
        self._current_label  = None
        self._hold_counter   = 0
        self._cooldown_left  = 0
        self._last_word      = None
        self._sentence_ready = False
