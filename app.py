"""
app.py
------
AI-Based Bidirectional Sign Language Communication System
WITH:
  - Regional Language Support (Hindi/Tamil/Telugu/Marathi etc.)
  - Live Chat Mode (split screen - both people on same screen)
  - Sign Animations (animated sign display for hearing person)
  - Groq AI Assistant (sentence improvement, suggestions, full chat)

Run with:
    streamlit run app.py
"""

import os
import sys
import time
import numpy as np
import cv2
import importlib
try:
    dotenv = importlib.import_module("dotenv")
    load_dotenv = dotenv.load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False
load_dotenv()
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sign Language AI",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@600&display=swap');

    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --accent: #f59e0b;
        --success: #10b981;
        --bg-dark: #0f172a;
        --glass-bg: rgba(30, 41, 59, 0.7);
        --glass-border: rgba(255, 255, 255, 0.1);
    }

    .main {
        background-color: var(--bg-dark);
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top right, #1e1b4b, #0f172a);
    }

    .gesture-badge {
        display: inline-block;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        color: white;
        padding: 0.5rem 1.25rem;
        border-radius: 12px;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0.3rem;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        font-family: 'Poppins', sans-serif;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .sentence-box {
        background: var(--glass-bg);
        backdrop-filter: blur(12px);
        border: 1px solid var(--glass-border);
        border-radius: 16px;
        padding: 1.5rem;
        font-size: 1.4rem;
        color: #f1f5f9;
        min-height: 80px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        margin: 1rem 0;
        line-height: 1.6;
    }

    .ai-box {
        background: rgba(13, 148, 136, 0.1);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(20, 184, 166, 0.3);
        border-radius: 16px;
        padding: 1.25rem;
        font-size: 1.2rem;
        color: #2dd4bf;
        min-height: 60px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        margin: 0.5rem 0;
    }

    /* ── Live Chat Styles ─────────────────────────────────────────── */
    .lc-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0 0 1rem 0;
    }
    .lc-status {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #10b981;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .lc-dot {
        width: 9px; height: 9px;
        background: #10b981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10b981;
        animation: blink 2s infinite;
    }
    @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.4} }

    .lc-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 1.2rem 1.3rem;
        color: #1e293b;
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    .lc-card-head {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1rem;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid #f1f5f9;
    }
    .lc-avatar {
        width: 40px; height: 40px;
        border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 13px;
    }
    .av-green { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
    .av-blue  { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }
    .lc-badge {
        margin-left: auto;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 700;
    }
    .badge-green { background: #ecfdf5; color: #059669; }
    .badge-blue  { background: #eff6ff; color: #2563eb; }

    .sign-preview-wrap {
        background: #fafaf9;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 8px 10px;
        margin: 8px 0;
    }
    .sign-preview-label {
        font-size: 10px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .sign-tiles-row {
        display: flex;
        flex-wrap: wrap;
        gap: 5px;
    }
    .sign-tile {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 36px;
        padding: 5px 3px 3px;
        border-radius: 7px;
        font-size: 18px;
        line-height: 1;
    }
    .sign-tile-letter {
        font-size: 9px;
        font-weight: 700;
        margin-top: 2px;
    }
    .tile-green { background: #ecfdf5; color: #059669; }
    .tile-blue  { background: #dbeafe; color: #1d4ed8; }

    .cam-activate {
        border: 1.5px dashed #d1d5db;
        border-radius: 10px;
        padding: 22px;
        text-align: center;
        cursor: pointer;
        color: #9ca3af;
        font-size: 13px;
        margin-bottom: 8px;
        transition: all 0.15s;
        background: #fafaf9;
    }
    .cam-activate:hover { border-color: #059669; color: #059669; }

    /* ── Chat Message Styles ──────────────────────────────────────── */
    .chat-stream {
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 4px 0;
    }
    .chat-msg { display: flex; flex-direction: column; max-width: 88%; }
    .chat-msg-a { align-self: flex-start; }
    .chat-msg-b { align-self: flex-end; align-items: flex-end; }
    .chat-sender {
        font-size: 10px;
        color: #94a3b8;
        font-weight: 600;
        margin-bottom: 3px;
    }
    .chat-bubble {
        padding: 8px 13px;
        border-radius: 14px;
        font-size: 14px;
        line-height: 1.5;
        color: #1e293b;
    }
    .cbubble-a {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-bottom-left-radius: 3px;
    }
    .cbubble-b {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-bottom-right-radius: 3px;
    }
    .chat-signs-wrap {
        margin-top: 5px;
        padding: 6px 9px;
        border-radius: 9px;
    }
    .csigns-a { background: #f0fdf4; border: 1px solid #a7f3d0; }
    .csigns-b { background: #eff6ff; border: 1px solid #bfdbfe; }
    .csigns-label {
        font-size: 9px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        margin-bottom: 5px;
    }
    .clabel-a { color: #059669; }
    .clabel-b { color: #2563eb; }
    .chat-type-tag {
        font-size: 10px;
        color: #cbd5e1;
        margin-top: 3px;
        display: flex;
        align-items: center;
        gap: 4px;
    }
    .wave-wrap {
        display: inline-flex;
        align-items: center;
        gap: 2px;
        height: 14px;
        vertical-align: middle;
    }
    .wbar {
        width: 2px;
        border-radius: 1px;
        background: #2563eb;
        display: inline-block;
    }

    .lc-footer {
        margin-top: 1.5rem;
        padding: 0.8rem 1rem;
        background: rgba(255,255,255,0.04);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 12px;
        color: #475569;
        font-size: 0.8rem;
        border: 1px solid rgba(255,255,255,0.06);
    }

    /* ── General Streamlit Overrides ──────────────────────────────── */
    h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700 !important;
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    .stButton > button {
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        border: none !important;
        background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 20px rgba(79, 70, 229, 0.4) !important;
        background: linear-gradient(135deg, #4f46e5, #4338ca) !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(30, 41, 59, 0.5);
        padding: 8px;
        border-radius: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 10px;
        background-color: transparent;
        border: none;
        color: #94a3b8;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(255, 255, 255, 0.05);
        color: #f8fafc;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: white !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(to right, #6366f1, #a855f7);
    }
</style>
""", unsafe_allow_html=True)


# ── Language map ───────────────────────────────────────────────────────────────
LANG_MAP = {
    "English":            "en",
    "Hindi (हिन्दी)":     "hi",
    "Tamil (தமிழ்)":      "ta",
    "Telugu (తెలుగు)":    "te",
    "Marathi (मराठी)":    "mr",
    "Bengali (বাংলা)":    "bn",
    "Gujarati (ગુજરાતી)": "gu",
    "Kannada (ಕನ್ನಡ)":   "kn",
    "Malayalam (മലയാളം)": "ml",
    "Punjabi (ਪੰਜਾਬੀ)":  "pa",
}

# ── Sign Language Hand Map ─────────────────────────────────────────────────────
SIGN_HAND_MAP = {
    'A': '🤙', 'B': '✌️', 'C': '🤏', 'D': '👆', 'E': '✊',
    'F': '🤘', 'G': '👉', 'H': '🤞', 'I': '🤙', 'J': '👈',
    'K': '✌️', 'L': '👋', 'M': '✊', 'N': '🤚', 'O': '👌',
    'P': '👇', 'Q': '👆', 'R': '🤞', 'S': '✊', 'T': '👍',
    'U': '✌️', 'V': '✌️', 'W': '🖐️', 'X': '☝️', 'Y': '🤙', 'Z': '👆',
}

# ── Session State ──────────────────────────────────────────────────────────────
_defaults = {
    "sentence":           "",
    "words":              [],
    "last_gesture":       "",
    "confidence":         0.0,
    "camera_running":     False,
    "frame_count":        0,
    "translated_text":    "",
    "ai_improved":        "",
    "ai_suggestions":     [],
    "groq_assistant":     None,
    "chat_log":           [],
    "anim_sentence":      "",
    "sign_grid":          None,
    "sign_text_input":    "",
    "animator":           None,
    "ai_chat_messages":   [],
    "lc_a_mode":          "sign",
    "lc_b_mode":          "voice",
    "lc_sign_preview":    "",
    "lc_a_cam_active":    False,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helper functions ───────────────────────────────────────────────────────────
def render_sign_tiles_html(text: str, color: str = "green", max_chars: int = 20) -> str:
    tile_cls = "tile-green" if color == "green" else "tile-blue"
    letter_color = "#059669" if color == "green" else "#1d4ed8"
    tiles = ""
    count = 0
    for ch in text.upper():
        if count >= max_chars:
            tiles += '<div class="sign-tile tile-green" style="background:#f1f5f9">…</div>'
            break
        if ch == ' ':
            tiles += '<div style="width:8px"></div>'
            continue
        emoji = SIGN_HAND_MAP.get(ch, '🤟')
        tiles += (
            f'<div class="sign-tile {tile_cls}">'
            f'  {emoji}'
            f'  <span class="sign-tile-letter" style="color:{letter_color}">{ch}</span>'
            f'</div>'
        )
        count += 1
    return f'<div class="sign-tiles-row">{tiles}</div>'


def build_chat_message_html(msg: dict) -> str:
    is_a   = msg["role"] == "mute"
    align  = "chat-msg-a" if is_a else "chat-msg-b"
    bubble = "cbubble-a"  if is_a else "cbubble-b"
    swrap  = "csigns-a"   if is_a else "csigns-b"
    slabel = "clabel-a"   if is_a else "clabel-b"
    color  = "green"      if is_a else "blue"
    sender = "✋ Person A (Mute)" if is_a else "🗣️ Person B (Speaker)"
    text   = msg.get("text", "")
    itype  = msg.get("input_type", "typed")

    if itype == "sign":
        type_tag = "✋ sign language"
    elif itype == "voice":
        bars = "".join(
            f'<span class="wbar" style="height:{h}px"></span>'
            for h in [4, 9, 13, 7, 11, 5, 8]
        )
        type_tag = f'🎙️ voice &nbsp;<span class="wave-wrap">{bars}</span>'
    else:
        type_tag = "⌨️ typed"

    sign_label = "Sign language:" if is_a else "Signs shown to Person A:"
    sign_tiles = render_sign_tiles_html(text, color=color, max_chars=22)

    return f"""
    <div class="chat-msg {align}">
        <div class="chat-sender">{sender}</div>
        <div class="chat-bubble {bubble}">{text}</div>
        <div class="chat-signs-wrap {swrap}">
            <div class="csigns-label {slabel}">{sign_label}</div>
            {sign_tiles}
        </div>
        <div class="chat-type-tag">{type_tag}</div>
    </div>
    """


# ── Resource loaders ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model_resources():
    res = {"recognizer": None, "tts": None, "error": None}
    try:
        from modules.gesture_recognizer import GestureRecognizer
        res["recognizer"] = GestureRecognizer()
    except (FileNotFoundError, ModuleNotFoundError) as e:
        res["error"] = f"Model not found: {e}\nRun `python train_model.py` first."
        return res
    except Exception as e:
        res["error"] = f"Failed to load model: {e}"
        return res
    try:
        from modules.tts_engine import TTSEngine
        res["tts"] = TTSEngine(engine="auto")
    except Exception:
        pass
    return res


def get_groq(api_key, model):
    if st.session_state.groq_assistant is None and api_key:
        try:
            from modules.groq_assistant import GroqAssistant
            st.session_state.groq_assistant = GroqAssistant(api_key=api_key, model=model)
        except Exception as e:
            st.sidebar.error(f"Groq error: {e}")
    return st.session_state.groq_assistant


def get_animator():
    if st.session_state.animator is None:
        try:
            from modules.sign_animator import SignAnimator
            st.session_state.animator = SignAnimator(frame_size=(280, 280), fps=1.2)
        except Exception as e:
            st.error(f"Sign Animator error: {e}")
            return None
    return st.session_state.animator


def speak_text(tts, text, lang="en"):
    if not tts or not text.strip():
        return
    try:
        if lang == "en":
            tts.speak(text, blocking=False)
        else:
            from gtts import gTTS
            import tempfile, pygame
            obj = gTTS(text=text, lang=lang, slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                obj.save(f.name)
                tmp = f.name
            pygame.mixer.init()
            pygame.mixer.music.load(tmp)
            pygame.mixer.music.play()
    except Exception as e:
        st.toast(f"TTS error: {e}", icon="⚠️")


def render_compact_sentence_box(text, placeholder):
    display_text = text.strip() if text and text.strip() else placeholder
    st.markdown(
        '<div class="sentence-box" style="font-size:1rem; min-height:55px; padding:10px;">'
        f'{display_text}</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <img src="https://img.icons8.com/fluency/96/hand-with-pen.png" width="80">
            <h2 style="margin-top: 10px; color: white !important;">Control Panel</h2>
        </div>
    """, unsafe_allow_html=True)
    st.divider()

    st.subheader("🤖 Intelligence")
    groq_api_key = st.text_input(
        "Groq API Key", type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        placeholder="gsk_...",
        help="Free key at https://console.groq.com",
    )
    groq_model = st.selectbox("Model", [
        "llama-3.1-8b-instant", "llama-3.3-70b-versatile",
        "mixtral-8x7b-32768", "gemma2-9b-it",
    ])
    use_ai_improve = st.checkbox("Auto-improve sentences", value=True)
    use_ai_suggest = st.checkbox("Show word suggestions",  value=True)

    st.divider()
    st.subheader("🌐 Output Language")
    output_lang_name = st.selectbox("Language", list(LANG_MAP.keys()))
    output_lang_code = LANG_MAP[output_lang_name]

    st.divider()
    st.subheader("🎛️ Detection")
    conf_thresh  = st.slider("Confidence",      0.50, 0.99, 0.75, 0.01)
    hold_frames  = st.slider("Hold Frames",     5,  40, 15, 1)
    cooldown_frm = st.slider("Cooldown Frames", 10, 60, 30, 1)

    st.divider()
    st.subheader("📷 Camera")
    camera_index  = st.number_input("Camera Index", 0, 5, 0)
    flip_camera   = st.checkbox("Flip Camera",       value=True)
    show_fps      = st.checkbox("Show FPS",          value=True)
    speak_on_word = st.checkbox("Speak each word",   value=False)
    speak_on_done = st.checkbox("Speak full sentence", value=True)

    st.divider()
    st.caption("🤟 Made for Deaf & Mute Community")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
    <div style="text-align: center; padding: 2rem 0; margin-bottom: 2rem;
         background: rgba(255,255,255,0.03); border-radius: 24px;
         border: 1px solid rgba(255,255,255,0.05);">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">🤟 SignBridge AI</h1>
        <p style="font-size: 1.2rem; color: #94a3b8; max-width: 600px; margin: 0 auto;">
            Empowering the Deaf and Mute community with real-time bidirectional
            AI sign language translation and regional language support.
        </p>
    </div>
""", unsafe_allow_html=True)

# ── Load resources ─────────────────────────────────────────────────────────────
with st.spinner("Loading AI model..."):
    res = load_model_resources()
if res.get("error"):
    st.error(res["error"])
    st.stop()

recognizer = res["recognizer"]
tts        = res.get("tts")
groq       = get_groq(groq_api_key, groq_model) if groq_api_key else None
animator   = get_animator()


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📷 Sign → Text/Voice",
    "🌐 Regional Language",
    "💬 Live Chat",
    "🎬 Text → Sign Animation",
    "🤖 AI Assistant",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — Sign → Text/Voice
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    col_cam, col_info = st.columns([3, 2], gap="large")

    with col_cam:
        st.subheader("📷 Live Camera Feed")
        cam_ph = st.empty()
        b1, b2, b3 = st.columns(3)
        start_btn = b1.button("▶️ Start",  use_container_width=True, type="primary")
        stop_btn  = b2.button("⏹️ Stop",   use_container_width=True)
        clear_btn = b3.button("🗑️ Clear",  use_container_width=True)

    with col_info:
        st.subheader("🔍 Output")

        st.markdown("**Detected Gesture:**")
        gest_ph = st.empty()
        gest_ph.markdown(
            f'<span class="gesture-badge">'
            f'{st.session_state.last_gesture or "Waiting..."}</span>',
            unsafe_allow_html=True,
        )

        st.markdown("**Confidence:**")
        conf_ph = st.empty()
        conf_ph.progress(float(st.session_state.confidence))

        st.markdown("---")
        st.markdown("**📝 Sentence (English):**")
        sent_ph = st.empty()
        _st = st.session_state.sentence or '<i style="color:#666">Waiting...</i>'
        sent_ph.markdown(f'<div class="sentence-box">{_st}</div>', unsafe_allow_html=True)

        if groq and st.session_state.ai_improved:
            st.markdown("**🤖 AI Improved:**")
            st.markdown(
                f'<div class="ai-box">{st.session_state.ai_improved}</div>',
                unsafe_allow_html=True,
            )

        if groq and use_ai_suggest and st.session_state.ai_suggestions:
            st.markdown("**💡 Suggestions:**")
            sc = st.columns(3)
            for i, sg in enumerate(st.session_state.ai_suggestions[:3]):
                if sc[i].button(sg, key=f"sg{i}", use_container_width=True):
                    st.session_state.words.append(sg)
                    st.session_state.sentence = " ".join(st.session_state.words)
                    st.rerun()

        st.markdown("---")
        sp_c, un_c = st.columns(2)
        speak_btn = sp_c.button("🔊 Speak",     use_container_width=True)
        undo_btn  = un_c.button("↩️ Undo Word", use_container_width=True)

        if speak_btn:
            txt = st.session_state.ai_improved or st.session_state.sentence
            speak_text(tts, txt, output_lang_code)
            st.toast("Speaking...", icon="🔊")

        if undo_btn and st.session_state.words:
            st.session_state.words.pop()
            st.session_state.sentence = " ".join(st.session_state.words)
            st.rerun()

        st.markdown("**🏷️ Word History:**")
        hist_ph = st.empty()
        if st.session_state.words:
            badges = "".join(
                f'<span class="gesture-badge">{w}</span>'
                for w in st.session_state.words[-10:]
            )
            hist_ph.markdown(badges, unsafe_allow_html=True)
        else:
            hist_ph.caption("No words yet")

    if start_btn:
        st.session_state.camera_running = True
    if stop_btn:
        st.session_state.camera_running = False
    if clear_btn:
        st.session_state.update({
            "sentence": "", "words": [], "last_gesture": "",
            "confidence": 0.0, "ai_improved": "", "ai_suggestions": [],
        })
        st.rerun()

    if st.session_state.camera_running:
        from modules.hand_detector    import HandDetector
        from modules.sentence_builder import SentenceBuilder

        hd  = HandDetector(min_detection_confidence=conf_thresh)
        sb  = SentenceBuilder(hold_frames=hold_frames, cooldown_frames=cooldown_frm)
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            st.error(f"Cannot open camera {camera_index}.")
            st.session_state.camera_running = False
        else:
            prev_t = time.time()
            ai_t   = time.time()

            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    break
                if flip_camera:
                    frame = cv2.flip(frame, 1)

                annotated, landmarks, hand_found = hd.process(frame)
                label, conf = None, 0.0
                if hand_found and landmarks is not None:
                    label, conf = recognizer.predict(landmarks)

                new_word = sb.update(label)
                if new_word:
                    st.session_state.words.append(new_word)
                    st.session_state.sentence = " ".join(st.session_state.words)
                    if speak_on_word:
                        speak_text(tts, new_word, output_lang_code)
                    if groq and time.time() - ai_t > 2.0:
                        ai_t = time.time()
                        if use_ai_improve:
                            st.session_state.ai_improved = groq.improve_sentence(
                                st.session_state.sentence
                            )
                        if use_ai_suggest:
                            st.session_state.ai_suggestions = groq.suggest_next(
                                st.session_state.sentence
                            )

                st.session_state.last_gesture = label or ""
                st.session_state.confidence   = conf

                curr_t = time.time()
                fps    = 1 / max(curr_t - prev_t, 1e-6)
                prev_t = curr_t

                if show_fps:
                    cv2.putText(annotated, f"FPS:{fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 150), 2)
                cv2.putText(
                    annotated,
                    f"Gesture: {label or 'None'}",
                    (10, annotated.shape[0] - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 100) if label else (100, 100, 255), 2,
                )

                frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                cam_ph.image(frame_rgb, channels="RGB", use_container_width=True)
                gest_ph.markdown(
                    f'<span class="gesture-badge">{label or "Detecting..."}</span>',
                    unsafe_allow_html=True,
                )
                conf_ph.progress(min(float(conf), 1.0))
                _s = st.session_state.sentence or '<i style="color:#666">Waiting...</i>'
                sent_ph.markdown(f'<div class="sentence-box">{_s}</div>', unsafe_allow_html=True)
                if st.session_state.words:
                    hist_ph.markdown(
                        "".join(
                            f'<span class="gesture-badge">{w}</span>'
                            for w in st.session_state.words[-10:]
                        ),
                        unsafe_allow_html=True,
                    )

                st.session_state.frame_count += 1

                if sb.is_sentence_ready():
                    final_txt = st.session_state.ai_improved or st.session_state.sentence
                    if final_txt and speak_on_done and tts:
                        speak_text(tts, final_txt, output_lang_code)
                    sb.clear()

                time.sleep(0.01)

            cap.release()
            hd.close()


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — Regional Language Output
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("🌐 Regional Language Output")

    curr_sentence = st.session_state.sentence
    st.markdown("**Detected Sentence (English):**")
    st.markdown(
        f'<div class="sentence-box">'
        f'{curr_sentence or "No sentence yet — use Tab 1 first."}</div>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    sel_lang_name = st.selectbox("Translate to:", list(LANG_MAP.keys()), key="t2_lang")
    sel_lang_code = LANG_MAP[sel_lang_name]

    tc1, tc2 = st.columns(2)
    tr_btn = tc1.button("🔄 Translate",        use_container_width=True, type="primary")
    sp_tr  = tc2.button("🔊 Speak Translation", use_container_width=True)

    if tr_btn:
        if not curr_sentence:
            st.warning("No sentence yet. Use Tab 1 first.")
        elif not groq:
            st.error("Enter Groq API key in sidebar.")
        else:
            with st.spinner("Translating..."):
                st.session_state.translated_text = groq.translate(curr_sentence, sel_lang_code)

    if st.session_state.translated_text:
        st.markdown(f"**{sel_lang_name} Translation:**")
        st.markdown(
            f'<div class="ai-box" style="font-size:1.4rem;">'
            f'{st.session_state.translated_text}</div>',
            unsafe_allow_html=True,
        )
        if sp_tr:
            speak_text(tts, st.session_state.translated_text, sel_lang_code)
            st.toast(f"Speaking in {sel_lang_name}!", icon="🔊")

    st.divider()
    st.subheader("✍️ Manual Translation")
    mt = st.text_area("Type any text:", height=80, key="mt_text")
    ml1, ml2 = st.columns([2, 1])
    ml_to  = ml1.selectbox("To:", list(LANG_MAP.keys()), key="ml_to")
    ml_btn = ml2.button("Translate", use_container_width=True, key="ml_btn")
    if ml_btn and mt.strip():
        if not groq:
            st.error("Enter Groq API key in sidebar.")
        else:
            with st.spinner("Translating..."):
                res_t = groq.translate(mt, LANG_MAP[ml_to])
            st.markdown(f'<div class="ai-box">{res_t}</div>', unsafe_allow_html=True)
            speak_text(tts, res_t, LANG_MAP[ml_to])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — Live Chat Mode
# ─────────────────────────────────────────────────────────────────────────────
with tab3:

    # ── process any pending send BEFORE rendering (avoids slow rerun) ─────────
    _pending = st.session_state.pop("_pending_msg", None)
    if _pending:
        st.session_state.chat_log.append(_pending)
        if _pending["role"] == "mute":
            st.session_state.sentence    = ""
            st.session_state.words       = []
            st.session_state.ai_improved = ""

    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown("""
        <div class="lc-header">
            <div>
                <h2 style="margin:0;color:white!important;font-family:'Poppins',sans-serif;">
                    Live communication
                </h2>
                <p style="margin:0;color:#94a3b8;font-size:0.88rem;">
                    Sign language + Voice bridge — real-time
                </p>
            </div>
            <div class="lc-status">
                <div class="lc-dot"></div> Connected
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Conversation stream (pure HTML — renders instantly, no container bug) ─
    st.markdown("#### 🗨️ Conversation Stream")
    if not st.session_state.chat_log:
        st.markdown("""
            <div style="text-align:center;padding:50px 0;opacity:0.45;
                 background:rgba(255,255,255,0.02);border-radius:16px;
                 border:1px solid rgba(255,255,255,0.06);margin-bottom:1rem;">
                <div style="font-size:2.2rem;">💬</div>
                <p style="color:#94a3b8;font-size:0.9rem;margin-top:8px;">
                    No messages yet — start communicating below!
                </p>
            </div>
        """, unsafe_allow_html=True)
    else:
        msgs_html = "".join(build_chat_message_html(m) for m in st.session_state.chat_log)
        st.markdown(f"""
            <div style="
                max-height:360px; overflow-y:auto;
                padding:14px 16px;
                background:rgba(255,255,255,0.02);
                border-radius:16px;
                border:1px solid rgba(255,255,255,0.06);
                margin-bottom:1rem;
            ">
                <div class="chat-stream">{msgs_html}</div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()

    col_a, col_b = st.columns(2, gap="large")

    # ════════════════════════════════
    #  PERSON A — Mute
    # ════════════════════════════════
    with col_a:
        st.markdown("""
            <div class="lc-card">
                <div class="lc-card-head">
                    <div class="lc-avatar av-green">MA</div>
                    <div>
                        <div style="font-weight:700;font-size:14px;color:#1e293b;">
                            Person A — Mute
                        </div>
                        <div style="font-size:11px;color:#64748b;">Uses sign language</div>
                    </div>
                    <span class="lc-badge badge-green">Sign</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        a_mode = st.radio(
            "A input mode",
            ["✋  Hand Sign", "⌨️  Type"],
            horizontal=True,
            key="a_input_mode_tab3",
            label_visibility="collapsed",
        )

        if "Hand Sign" in a_mode:
            current_sign_text = st.session_state.sentence
            if current_sign_text:
                st.markdown(f"""
                    <div class="sign-preview-wrap">
                        <div class="sign-preview-label">✋ Currently signing — "{current_sign_text}"</div>
                        {render_sign_tiles_html(current_sign_text, color="green")}
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="cam-activate">
                        <div style="font-size:1.6rem;margin-bottom:4px;">✋</div>
                        Activate camera in Tab 1 to start signing
                    </div>
                """, unsafe_allow_html=True)

            sign_word_input = st.text_input(
                "Or type a word to sign:",
                key="lc_sign_word_inp",
                placeholder="e.g. HELLO",
            )
            if sign_word_input:
                st.markdown(f"""
                    <div class="sign-preview-wrap" style="margin-top:4px;">
                        <div class="sign-preview-label">Sign preview:</div>
                        {render_sign_tiles_html(sign_word_input, color="green")}
                    </div>
                """, unsafe_allow_html=True)

            if st.button("📤 Send Sign Message", use_container_width=True,
                         type="primary", key="send_a_sign_v4"):
                _txt = (
                    sign_word_input.strip()
                    or (st.session_state.ai_improved or st.session_state.sentence).strip()
                )
                if _txt:
                    st.session_state["_pending_msg"] = {
                        "role": "mute", "text": _txt, "input_type": "sign"
                    }
                    st.toast(f"✅ Sent!", icon="✋")
                    st.rerun()
                else:
                    st.warning("Nothing to send — sign something or type a word above.")

        else:
            a_type_text = st.text_area(
                "Type your message:",
                key="lc_a_type_text",
                height=90,
                placeholder="Type what you want to communicate...",
            )
            if a_type_text.strip():
                st.markdown(f"""
                    <div class="sign-preview-wrap">
                        <div class="sign-preview-label">Live sign preview:</div>
                        {render_sign_tiles_html(a_type_text, color="green")}
                    </div>
                """, unsafe_allow_html=True)

            if st.button("📤 Send Text (A)", use_container_width=True,
                         type="primary", key="send_a_text_v4"):
                if a_type_text.strip():
                    st.session_state["_pending_msg"] = {
                        "role": "mute", "text": a_type_text.strip(), "input_type": "typed"
                    }
                    st.toast("✅ Sent!", icon="✋")
                    st.rerun()
                else:
                    st.warning("Please type a message first.")

        # last message received from B
        last_from_b = next(
            (m for m in reversed(st.session_state.chat_log) if m["role"] == "speaker"), None
        )
        if last_from_b:
            st.markdown("---")
            st.markdown(
                '<div style="font-size:11px;color:#64748b;margin-bottom:4px;">'
                '📩 Last received from Person B:</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"""
                <div style="background:#eff6ff;border:1px solid #bfdbfe;
                     border-radius:10px;padding:8px 12px;font-size:13px;color:#1e293b;">
                    {last_from_b["text"]}
                </div>
                <div class="sign-preview-wrap" style="margin-top:5px;">
                    <div class="sign-preview-label">Shown as signs:</div>
                    {render_sign_tiles_html(last_from_b["text"], color="blue")}
                </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════
    #  PERSON B — Speaker
    # ════════════════════════════════
    with col_b:
        import speech_recognition as sr

        st.markdown("""
            <div class="lc-card">
                <div class="lc-card-head">
                    <div class="lc-avatar av-blue">PB</div>
                    <div>
                        <div style="font-weight:700;font-size:14px;color:#1e293b;">
                            Person B — Speaker
                        </div>
                        <div style="font-size:11px;color:#64748b;">
                            Uses voice, no sign knowledge
                        </div>
                    </div>
                    <span class="lc-badge badge-blue">Voice</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        b_mode = st.radio(
            "B input mode",
            ["🎙️  Voice Record", "⌨️  Type"],
            horizontal=True,
            key="b_input_mode_tab3",
            label_visibility="collapsed",
        )

        if "Voice" in b_mode:
            if st.button(
                "🎙️  Click to Record Voice",
                use_container_width=True,
                type="primary",
                key="voice_rec_btn_v4",
            ):
                with st.status("🎤 Listening...", expanded=True) as status:
                    r = sr.Recognizer()
                    try:
                        with sr.Microphone() as source:
                            r.adjust_for_ambient_noise(source, duration=0.3)
                            status.update(label="🔴 Recording — speak now...", state="running")
                            audio = r.listen(source, timeout=5, phrase_time_limit=10)
                            status.update(label="⌛ Processing...", state="running")
                            transcript = r.recognize_google(audio)
                            st.session_state["_pending_msg"] = {
                                "role": "speaker", "text": transcript, "input_type": "voice"
                            }
                            status.update(label=f'✅ Heard: "{transcript}"', state="complete")
                            st.rerun()
                    except sr.UnknownValueError:
                        status.update(label="❌ Could not understand audio", state="error")
                    except sr.RequestError:
                        status.update(label="❌ Speech service unavailable", state="error")
                    except Exception as e:
                        status.update(label=f"❌ Error: {e}", state="error")

            st.markdown(
                '<div style="text-align:center;color:#94a3b8;font-size:11px;margin:8px 0;">'
                '— or type below —</div>',
                unsafe_allow_html=True,
            )
            b_voice_type = st.text_input(
                "Type message (B):",
                key="b_voice_type_inp",
                placeholder="Type something...",
                label_visibility="collapsed",
            )
            if b_voice_type.strip():
                st.markdown(f"""
                    <div class="sign-preview-wrap">
                        <div class="sign-preview-label">Signs shown to Person A:</div>
                        {render_sign_tiles_html(b_voice_type, color="blue")}
                    </div>
                """, unsafe_allow_html=True)

            if st.button("📤 Send (Voice/Type)", use_container_width=True, key="send_b_vt_v4"):
                if b_voice_type.strip():
                    st.session_state["_pending_msg"] = {
                        "role": "speaker", "text": b_voice_type.strip(), "input_type": "typed"
                    }
                    st.toast("✅ Sent!", icon="🗣️")
                    st.rerun()
                else:
                    st.warning("Type a message or use voice recording.")

        else:
            b_type_text = st.text_area(
                "Type your message:",
                key="lc_b_type_text",
                height=90,
                placeholder="Type what you want to say to Person A...",
            )
            if b_type_text.strip():
                st.markdown(f"""
                    <div class="sign-preview-wrap">
                        <div class="sign-preview-label">Signs that will be shown to Person A:</div>
                        {render_sign_tiles_html(b_type_text, color="blue")}
                    </div>
                """, unsafe_allow_html=True)

            if st.button("📤 Send Text (B)", use_container_width=True,
                         type="primary", key="send_b_text_v4"):
                if b_type_text.strip():
                    st.session_state["_pending_msg"] = {
                        "role": "speaker", "text": b_type_text.strip(), "input_type": "typed"
                    }
                    st.toast("✅ Sent!", icon="🗣️")
                    st.rerun()
                else:
                    st.warning("Please type a message first.")

        # last message received from A
        last_from_a = next(
            (m for m in reversed(st.session_state.chat_log) if m["role"] == "mute"), None
        )
        if last_from_a:
            st.markdown("---")
            st.markdown(
                '<div style="font-size:11px;color:#64748b;margin-bottom:4px;">'
                '📩 Last received from Person A:</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"""
                <div style="background:#f0fdf4;border:1px solid #bbf7d0;
                     border-radius:10px;padding:8px 12px;font-size:13px;color:#1e293b;">
                    {last_from_a["text"]}
                </div>
                <div style="font-size:10px;color:#94a3b8;margin-top:3px;">
                    ✋ via {last_from_a.get("input_type","sign")}
                </div>
            """, unsafe_allow_html=True)

    # ── Footer + Reset ────────────────────────────────────────────────────────
    st.markdown("""
        <div class="lc-footer">
            ✋ Sign/Type (Person A) &nbsp;↔&nbsp; 🎙️ Voice/Type (Person B)
            &nbsp;|&nbsp; Both see sign hands for every message
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    if st.button("🗑️ Reset Conversation", use_container_width=True, key="reset_chat_v4"):
        st.session_state.chat_log    = []
        st.session_state.sentence    = ""
        st.session_state.words       = []
        st.session_state.ai_improved = ""
        st.session_state.pop("_pending_msg", None)
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — Text → Sign Animation
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🎬 Text → Animated Sign Language")
    st.markdown(
        "Hearing person types → system shows each sign image one by one "
        "so the **mute person can read the signs**."
    )
    st.divider()

    if animator is None:
        st.error("Sign Animator could not be loaded. Check that `modules/sign_animator.py` exists.")
        st.stop()

    in_col, show_col = st.columns([1, 2], gap="large")

    with in_col:
        st.markdown("### Input")
        anim_text = st.text_area(
            "Enter sentence:",
            value=st.session_state.anim_sentence,
            height=110,
            placeholder="e.g. Hello how are you",
        )
        a1, a2   = st.columns(2)
        anim_btn = a1.button("🎬 Animate",   use_container_width=True, type="primary")
        grid_btn = a2.button("🔲 Show Grid", use_container_width=True)
        anim_fps = st.slider("Speed (signs/sec)", 0.5, 3.0, 1.2, 0.1)
        animator.frame_delay = 1.0 / anim_fps

        if anim_btn and anim_text.strip():
            st.session_state.anim_sentence = anim_text
            st.session_state.sign_grid     = None
            signs = animator.load_sentence(anim_text)
            st.success(f"Loaded {len(signs)} signs.")

        if grid_btn and anim_text.strip():
            animator.load_sentence(anim_text)
            grid = animator.build_grid(cols=5)
            st.session_state.sign_grid = grid

        if animator.signs:
            st.markdown("**Signs:**")
            for i, (lbl, path) in enumerate(animator.signs):
                st.caption(f"{'✅' if path else '⚠️'} {i+1}. {lbl}")

    with show_col:
        st.markdown("### 🤟 Sign Display")
        if st.session_state.sign_grid is not None:
            grid_rgb = cv2.cvtColor(st.session_state.sign_grid, cv2.COLOR_BGR2RGB)
            st.image(grid_rgb, caption="All signs", use_container_width=True)
            if st.button("▶️ Switch to Animation"):
                st.session_state.sign_grid = None
                st.rerun()
        elif animator.signs:
            all_frames   = animator.get_all_frames()
            anim_ph      = st.empty()
            label_ph     = st.empty()
            prog_ph      = st.empty()
            total_frames = len(all_frames)
            for _ in range(2):
                for idx, (lbl, frame) in enumerate(all_frames):
                    anim_ph.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), width=300)
                    label_ph.markdown(
                        f'<div style="text-align:center;">'
                        f'<span class="gesture-badge" style="font-size:1.5rem;">'
                        f'{lbl}</span></div>',
                        unsafe_allow_html=True,
                    )
                    prog_ph.progress(
                        (idx + 1) / total_frames,
                        text=f"Sign {idx+1} of {total_frames}",
                    )
                    time.sleep(1.0 / anim_fps)
            st.success("Animation complete!")
        else:
            st.info("Enter text and click **Animate** to begin.")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 — AI Assistant (Groq)
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("🤖 AI Assistant — Powered by Groq")

    if not groq:
        st.warning(
            "Enter your **Groq API Key** in the sidebar.\n\n"
            "Get a FREE key at: https://console.groq.com"
        )
        st.info(
            "With Groq AI you can:\n"
            "- Improve broken sign sentences automatically\n"
            "- Translate to any Indian regional language\n"
            "- Get word suggestions while signing\n"
            "- Have a full AI conversation\n"
            "- Ask AI to explain any sign gesture"
        )
    else:
        st.success(f"Groq AI connected — `{groq_model}`")
        st.divider()

        st.subheader("Quick Actions")
        qa1, qa2, qa3 = st.columns(3)

        if qa1.button("🔧 Improve Sentence", use_container_width=True):
            if st.session_state.sentence:
                with st.spinner("Improving..."):
                    imp = groq.improve_sentence(st.session_state.sentence)
                st.session_state.ai_improved = imp
                st.markdown(f'<div class="ai-box">{imp}</div>', unsafe_allow_html=True)
                speak_text(tts, imp, output_lang_code)
            else:
                st.warning("No sentence yet.")

        if qa2.button("💡 Suggest Next", use_container_width=True):
            if st.session_state.sentence:
                with st.spinner("Thinking..."):
                    sugs = groq.suggest_next(st.session_state.sentence)
                st.session_state.ai_suggestions = sugs
                cols = st.columns(3)
                for i, s in enumerate(sugs):
                    cols[i].markdown(
                        f'<span class="gesture-badge">{s}</span>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("No sentence yet.")

        if qa3.button("🔊 Speak Improved", use_container_width=True):
            txt = st.session_state.ai_improved or st.session_state.sentence
            if txt:
                speak_text(tts, txt, output_lang_code)
                st.toast("Speaking...", icon="🔊")

        st.divider()

        st.subheader("💬 AI Chat")
        ai_lang_sel  = st.selectbox("AI reply language:", list(LANG_MAP.keys()), key="ai_l")
        ai_lang_code = LANG_MAP[ai_lang_sel]

        with st.container(height=350):
            if not st.session_state.ai_chat_messages:
                st.caption("Ask anything...")
            for m in st.session_state.ai_chat_messages:
                if m["role"] == "user":
                    st.markdown(
                        f'<div class="chat-hear">👤 <b>You:</b> {m["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="chat-ai">🤖 <b>AI:</b> {m["content"]}</div>',
                        unsafe_allow_html=True,
                    )

        ai_input = st.text_input("Your message:", placeholder="Ask anything...", key="ai_inp")
        ai1, ai2 = st.columns([3, 1])
        ai_send  = ai1.button("Send",  use_container_width=True, type="primary")
        ai_clear = ai2.button("Clear", use_container_width=True)

        if ai_send and ai_input.strip():
            with st.spinner("Groq is thinking..."):
                reply = groq.chat(ai_input, language=ai_lang_code)
            st.session_state.ai_chat_messages += [
                {"role": "user",      "content": ai_input},
                {"role": "assistant", "content": reply},
            ]
            speak_text(tts, reply, ai_lang_code)
            st.rerun()

        if ai_clear:
            st.session_state.ai_chat_messages = []
            groq.clear_history()
            st.rerun()

        st.divider()

        st.subheader("🤟 Explain a Sign")
        e1, e2 = st.columns([2, 1])
        sign_q = e1.text_input("Which sign?", placeholder="e.g. Thank You", key="expl_q")
        if e2.button("Explain", use_container_width=True, key="expl_btn"):
            if sign_q:
                with st.spinner("Asking AI..."):
                    exp = groq.explain_sign(sign_q)
                st.info(exp)
