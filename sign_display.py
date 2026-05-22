"""
modules/sign_display.py
-----------------------
Text / Voice → Sign Language visual output.

Looks for sign images in:
    signs/words/<WORD>.jpg  (or .png)  – whole-word ISL sign images
    signs/alphabets/<LETTER>.jpg       – fingerspelling fallback

If an image is not found, a placeholder tile is rendered on the fly.

Voice input is handled via SpeechRecognition (microphone → text → signs).
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Tuple

try:
    import speech_recognition as sr
    _SR_AVAILABLE = True
except ImportError:
    _SR_AVAILABLE = False

from PIL import Image, ImageDraw, ImageFont


# ── Config ─────────────────────────────────────────────────────────────────────
WORDS_DIR     = "signs/words"
ALPHA_DIR     = "signs/alphabets"
TILE_SIZE     = (220, 220)          # each sign tile
THUMB_SIZE    = (180, 180)          # image within tile
COLS          = 4                   # tiles per row in the display grid
FONT_SIZE     = 18
BG_COLOR      = (30, 30, 30)
TILE_BG       = (50, 50, 60)
LABEL_COLOR   = (220, 220, 220)
BORDER_COLOR  = (80, 130, 200)
IMG_EXTS      = (".jpg", ".jpeg", ".png", ".bmp")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _find_image(directory: str, name: str) -> Optional[str]:
    """Return path to first matching image for `name` in `directory`."""
    for ext in IMG_EXTS:
        p = os.path.join(directory, f"{name}{ext}")
        if os.path.exists(p):
            return p
    return None


def _make_placeholder_tile(label: str) -> np.ndarray:
    """
    Create a simple placeholder image for gestures without a photo.
    Returns a (TILE_SIZE[1], TILE_SIZE[0], 3) uint8 numpy array.
    """
    tile = Image.new("RGB", TILE_SIZE, color=TILE_BG)
    draw = ImageDraw.Draw(tile)

    # Outer border
    draw.rectangle([2, 2, TILE_SIZE[0] - 3, TILE_SIZE[1] - 3],
                   outline=BORDER_COLOR, width=2)

    # Large label text in the centre
    try:
        # Try some common font paths for different OS
        font_paths = [
            "arialbd.ttf",  # Windows
            "DejaVuSans-Bold.ttf", # Linux
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", # Linux full path
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" # macOS
        ]
        font_large = None
        for path in font_paths:
            try:
                font_large = ImageFont.truetype(path, 32)
                break
            except OSError:
                continue
        
        if font_large is None:
            font_large = ImageFont.load_default()
            font_small = font_large
        else:
            font_small = ImageFont.truetype(font_large.path.replace("Bold", "").replace("bd", ""), 14)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = font_large

    # Abbreviate long words
    short = label[:8] + "…" if len(label) > 8 else label
    bbox  = draw.textbbox((0, 0), short, font=font_large)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((TILE_SIZE[0] - tw) // 2, (TILE_SIZE[1] - th) // 2 - 10),
              short, fill=(160, 200, 255), font=font_large)

    # Sub-label "No image"
    draw.text((10, TILE_SIZE[1] - 28), "[ no image ]",
              fill=(120, 120, 120), font=font_small)

    return np.array(tile)


def _load_tile(word: str) -> np.ndarray:
    """
    Load sign image for a word.  Falls back to placeholder if not found.
    """
    path = _find_image(WORDS_DIR, word.upper()) or _find_image(WORDS_DIR, word.lower())
    if path:
        img = cv2.imread(path)
        img = cv2.resize(img, THUMB_SIZE)
        # Pad to TILE_SIZE
        tile = np.full((*TILE_SIZE[::-1], 3), TILE_BG, dtype=np.uint8)
        oy   = (TILE_SIZE[1] - THUMB_SIZE[1]) // 2
        ox   = (TILE_SIZE[0] - THUMB_SIZE[0]) // 2
        tile[oy:oy+THUMB_SIZE[1], ox:ox+THUMB_SIZE[0]] = img

        # Label
        cv2.putText(tile, word.capitalize(),
                    (8, TILE_SIZE[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    LABEL_COLOR, 1, cv2.LINE_AA)
        return tile
    else:
        return _make_placeholder_tile(word)


def _word_to_tiles(word: str) -> List[np.ndarray]:
    """
    Try to find a whole-word sign; if not, fingerspell letter-by-letter.
    """
    path = (_find_image(WORDS_DIR, word.upper()) or
            _find_image(WORDS_DIR, word.lower()))
    if path:
        return [_load_tile(word)]

    # Fingerspell
    tiles = []
    for ch in word.upper():
        if ch == " " or not ch.isalpha():
            continue
        ch_path = _find_image(ALPHA_DIR, ch)
        if ch_path:
            img  = cv2.imread(ch_path)
            img  = cv2.resize(img, THUMB_SIZE)
            tile = np.full((*TILE_SIZE[::-1], 3), TILE_BG, dtype=np.uint8)
            oy   = (TILE_SIZE[1] - THUMB_SIZE[1]) // 2
            ox   = (TILE_SIZE[0] - THUMB_SIZE[0]) // 2
            tile[oy:oy+THUMB_SIZE[1], ox:ox+THUMB_SIZE[0]] = img
            cv2.putText(tile, ch, (8, TILE_SIZE[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        LABEL_COLOR, 1, cv2.LINE_AA)
            tiles.append(tile)
        else:
            tiles.append(_make_placeholder_tile(ch))
    return tiles


# ── Main class ─────────────────────────────────────────────────────────────────

class SignDisplay:
    """
    Converts a text string (or voice input) into a grid of sign images.

    Usage
    -----
        sd = SignDisplay()
        grid = sd.text_to_sign_grid("Hello Thank You")
        cv2.imshow("Signs", grid)
    """

    def __init__(self, cols: int = COLS):
        self.cols = cols

    # ── Text → Signs ──────────────────────────────────────────────────────────

    def text_to_tiles(self, text: str) -> List[np.ndarray]:
        """Convert full text sentence into a list of sign tiles."""
        words = text.strip().split()
        tiles: List[np.ndarray] = []
        for word in words:
            tiles.extend(_word_to_tiles(word))
        return tiles

    def text_to_sign_grid(self, text: str) -> np.ndarray:
        """
        Convert text to a displayable OpenCV image grid of sign tiles.
        Returns a numpy array (H, W, 3).
        """
        tiles = self.text_to_tiles(text)
        if not tiles:
            blank = np.full((TILE_SIZE[1], TILE_SIZE[0] * self.cols, 3),
                             BG_COLOR, dtype=np.uint8)
            cv2.putText(blank, "No text to display", (20, TILE_SIZE[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (180, 180, 180), 2)
            return blank

        return self._build_grid(tiles)

    # ── Voice → Signs ─────────────────────────────────────────────────────────

    def voice_to_sign_grid(
        self, timeout: int = 5, language: str = "en-IN"
    ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Listen to microphone, transcribe, and return (transcribed_text, grid).

        Returns (None, None) if speech recognition fails or is unavailable.
        """
        if not _SR_AVAILABLE:
            print("[SignDisplay] SpeechRecognition not installed.")
            return None, None

        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("[SignDisplay] Listening... (speak now)")
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                audio = recognizer.listen(source, timeout=timeout,
                                          phrase_time_limit=10)
            except sr.WaitTimeoutError:
                print("[SignDisplay] No speech detected.")
                return None, None

        try:
            text = recognizer.recognize_google(audio, language=language)
            print(f"[SignDisplay] Recognized: '{text}'")
            return text, self.text_to_sign_grid(text)
        except sr.UnknownValueError:
            print("[SignDisplay] Could not understand audio.")
            return None, None
        except sr.RequestError as e:
            print(f"[SignDisplay] SR service error: {e}")
            return None, None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_grid(self, tiles: List[np.ndarray]) -> np.ndarray:
        """Pack tiles into a multi-row grid image."""
        rows_needed = (len(tiles) + self.cols - 1) // self.cols
        tw, th = TILE_SIZE

        # Pad with blank tiles to fill last row
        while len(tiles) % self.cols != 0:
            tiles.append(np.full((th, tw, 3), BG_COLOR, dtype=np.uint8))

        rows = []
        for r in range(rows_needed):
            row_tiles = tiles[r * self.cols : (r + 1) * self.cols]
            rows.append(np.hstack(row_tiles))

        grid = np.vstack(rows)
        return grid
