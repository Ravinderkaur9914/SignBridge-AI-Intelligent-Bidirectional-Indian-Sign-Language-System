"""
modules/groq_assistant.py
--------------------------
AI Assistant powered by Groq API.
Helps mute person communicate better by:
- Improving broken sign-detected sentences
- Translating to regional Indian languages
- Suggesting next words
- Full AI chat mode

Install:
    pip install groq
"""

import importlib

try:
    groq_module = importlib.import_module("groq")
    Groq = groq_module.Groq
except ImportError as e:
    Groq = None
    _GROQ_IMPORT_ERROR = e

# ── Supported regional languages ──────────────────────────────────────────────
LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
    "bn": "Bengali",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
}


class GroqAssistant:
    """Groq-powered AI assistant for sign language communication."""

    def __init__(self, api_key: str, model: str = "llama-3.1-8b-instant"):
        if Groq is None:
            raise ImportError(
                "The groq package is required to use GroqAssistant. "
                "Install it with 'pip install groq'."
            ) from _GROQ_IMPORT_ERROR
        self.client  = Groq(api_key=api_key)
        self.model   = model
        self.history = []

        self.system_prompt = """You are a compassionate AI communication assistant 
helping a deaf/mute person communicate with hearing people in India.

Your roles:
1. IMPROVE  - take broken sign-language-style text and make it natural English
2. TRANSLATE - translate accurately to requested regional Indian languages
3. SUGGEST  - suggest what the person might want to say next
4. CHAT     - assist in two-way conversation between mute and hearing person

Always be:
- Short and clear (max 2-3 sentences)
- Respectful and empathetic  
- Culturally sensitive for Indian context

When translating, provide ONLY the translated text, nothing else."""

    # ── Improve a detected sentence ───────────────────────────────────────────
    def improve_sentence(self, raw_sentence: str) -> str:
        """
        Input:  'I help need doctor'   (raw sign detected)
        Output: 'I need help from a doctor.'  (natural sentence)
        """
        if not raw_sentence.strip():
            return ""
        prompt = (
            f'Convert this sign language detected text into a natural grammatically '
            f'correct sentence. Return ONLY the improved sentence.\n\n'
            f'Sign text: "{raw_sentence}"\nNatural sentence:'
        )
        return self._ask(prompt)

    # ── Translate to regional language ────────────────────────────────────────
    def translate(self, text: str, target_lang: str) -> str:
        """Translate text to Hindi / Tamil / Telugu etc."""
        if not text.strip():
            return ""
        lang_name = LANGUAGE_NAMES.get(target_lang, target_lang)
        prompt = (
            f"Translate this text to {lang_name}. "
            f"Return ONLY the translated text, no explanations.\n\n"
            f'Text: "{text}"\n{lang_name} translation:'
        )
        return self._ask(prompt)

    # ── Suggest next words ────────────────────────────────────────────────────
    def suggest_next(self, current_sentence: str) -> list:
        """Return 3 suggested next words/phrases as a list."""
        if not current_sentence.strip():
            return ["Hello", "Help", "Thank You"]
        prompt = (
            f'Given this partial sentence from a mute person: "{current_sentence}"\n'
            f"Suggest exactly 3 short words or phrases they might say next.\n"
            f"Return as comma-separated values ONLY. Example: Yes,Please,Help me\n"
            f"Suggestions:"
        )
        result = self._ask(prompt)
        parts = [s.strip() for s in result.split(",")][:3]
        return parts if len(parts) == 3 else ["Yes", "Please", "Thank You"]

    # ── Full AI chat ──────────────────────────────────────────────────────────
    def chat(self, user_message: str, language: str = "en") -> str:
        """
        Two-way chat. Maintains conversation history.
        Hearing person types → AI responds clearly.
        """
        lang_name = LANGUAGE_NAMES.get(language, "English")
        self.history.append({"role": "user", "content": user_message})

        # Keep last 10 messages only
        if len(self.history) > 10:
            self.history = self.history[-10:]

        messages = [{"role": "system", "content": self.system_prompt}]
        messages += self.history
        if language != "en":
            messages[-1]["content"] += f"\n\nPlease respond in {lang_name}."

        try:
            resp  = self.client.chat.completions.create(
                model=self.model, messages=messages,
                max_tokens=300, temperature=0.7,
            )
            reply = resp.choices[0].message.content.strip()
            self.history.append({"role": "assistant", "content": reply})
            return reply
        except Exception as e:
            return f"AI Error: {str(e)}"

    # ── Explain how to make a sign ────────────────────────────────────────────
    def explain_sign(self, gesture_name: str) -> str:
        prompt = (
            f'Briefly explain in 1-2 sentences how to make the Indian Sign Language '
            f'gesture for "{gesture_name}". Be simple and descriptive.'
        )
        return self._ask(prompt)

    def clear_history(self):
        self.history = []

    def _ask(self, prompt: str) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user",   "content": prompt},
                ],
                max_tokens=200, temperature=0.5,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            return f"Error: {str(e)}"