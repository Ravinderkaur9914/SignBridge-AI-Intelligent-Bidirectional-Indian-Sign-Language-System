"""
modules/tts_engine.py
----------------------
Text-to-Speech engine with auto-fallback:
  1. pyttsx3  (offline, fast, works on Windows/Linux/Mac)
  2. gTTS     (online, better Indian language support)
  3. Silent   (graceful no-op if neither available)

Install:
    pip install pyttsx3
    pip install gtts pygame        # for online fallback
"""

import threading


class TTSEngine:
    """
    Simple TTS wrapper. Use speak(text) to play audio.
    """

    def __init__(self, engine: str = "auto"):
        """
        engine : "pyttsx3" | "gtts" | "auto"
                 "auto" tries pyttsx3 first, falls back to gTTS.
        """
        self._engine_type = None
        self._engine      = None
        self._lock        = threading.Lock()

        if engine in ("auto", "pyttsx3"):
            self._try_init_pyttsx3()

        if self._engine is None and engine in ("auto", "gtts"):
            self._try_init_gtts()

        if self._engine is None:
            print("[TTSEngine] No TTS engine available — speech disabled.")

    # ── Init helpers ──────────────────────────────────────────────────────────
    def _try_init_pyttsx3(self):
        try:
            import pyttsx3
            eng = pyttsx3.init()
            eng.setProperty("rate",   150)
            eng.setProperty("volume", 1.0)
            self._engine      = eng
            self._engine_type = "pyttsx3"
        except Exception:
            pass

    def _try_init_gtts(self):
        try:
            from gtts import gTTS  # noqa: F401 — just check import
            self._engine_type = "gtts"
            self._engine      = True  # marker — gTTS has no persistent object
        except Exception:
            pass

    # ── Public API ────────────────────────────────────────────────────────────
    def speak(self, text: str, lang: str = "en", blocking: bool = False):
        """
        Speak `text` in `lang`.
        blocking=True waits until audio finishes.
        blocking=False runs in a background thread (default).
        """
        if not text or not text.strip() or self._engine is None:
            return

        if blocking:
            self._do_speak(text, lang)
        else:
            t = threading.Thread(target=self._do_speak, args=(text, lang),
                                 daemon=True)
            t.start()

    def _do_speak(self, text: str, lang: str):
        with self._lock:
            try:
                if self._engine_type == "pyttsx3":
                    self._engine.say(text)
                    self._engine.runAndWait()

                elif self._engine_type == "gtts":
                    import tempfile
                    import os
                    from gtts import gTTS
                    try:
                        import pygame
                        tts_obj = gTTS(text=text, lang=lang, slow=False)
                        with tempfile.NamedTemporaryFile(
                                suffix=".mp3", delete=False) as f:
                            tts_obj.save(f.name)
                            tmp = f.name
                        pygame.mixer.init()
                        pygame.mixer.music.load(tmp)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            pygame.time.Clock().tick(10)
                        os.unlink(tmp)
                    except Exception:
                        pass

            except Exception as e:
                print(f"[TTSEngine] speak error: {e}")

    def is_available(self) -> bool:
        return self._engine is not None
