import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

st.set_page_config(
    page_title="Video Aid",
    page_icon="▸",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --ink:       #0a0a0a;
    --ink-2:     #141414;
    --ink-3:     #1c1c1c;
    --ink-4:     #242424;
    --wire:      #2e2e2e;
    --wire-2:    #3a3a3a;
    --wire-3:    #505050;
    --smoke:     #787878;
    --ash:       #a0a0a0;
    --paper:     #d4d4d4;
    --white:     #f5f5f5;
    --lime:      #c8f135;
    --lime-dim:  rgba(200,241,53,0.08);
    --lime-mid:  rgba(200,241,53,0.16);
    --lime-glow: rgba(200,241,53,0.30);
    --red:       #ff4d4d;
    --red-dim:   rgba(255,77,77,0.10);
    --amber:     #ffb830;
    --amber-dim: rgba(255,184,48,0.10);
    --blue:      #4d9fff;
    --blue-dim:  rgba(77,159,255,0.10);
}

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
    background: var(--ink) !important;
    color: var(--white) !important;
    font-size: 15px;
    line-height: 1.6;
}
.stApp { background: var(--ink) !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ═══════════════════════════════════════════
   TOP NAV BAR
══════════════════════════════════════════════ */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2.5rem;
    height: 56px;
    border-bottom: 1px solid var(--wire);
    background: var(--ink);
    position: sticky; top: 0; z-index: 100;
}
.topbar-brand {
    display: flex; align-items: center; gap: 10px;
    font-family: 'Space Mono', monospace;
    font-size: 1rem; font-weight: 700;
    letter-spacing: -0.01em;
    color: var(--white);
}
.brand-mark {
    width: 28px; height: 28px;
    background: var(--lime);
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem; font-weight: 700;
    color: var(--ink);
    flex-shrink: 0;
}
.topbar-nav {
    display: flex; align-items: center; gap: 0.5rem;
}
.nav-tag {
    padding: 0.2rem 0.65rem;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: var(--ink-3);
    color: var(--smoke);
    border: 1px solid var(--wire);
}
.nav-tag.active {
    background: var(--lime-dim);
    color: var(--lime);
    border-color: rgba(200,241,53,0.25);
}
.status-live {
    display: flex; align-items: center; gap: 6px;
    font-size: 0.72rem; color: var(--smoke);
    letter-spacing: 0.06em; text-transform: uppercase;
}
.live-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--lime);
    box-shadow: 0 0 6px var(--lime-glow);
    animation: blink 2s infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

/* ═══════════════════════════════════════════
   MAIN LAYOUT: left panel + right panel
══════════════════════════════════════════════ */
.layout {
    display: grid;
    grid-template-columns: 340px 1fr;
    min-height: calc(100vh - 56px);
}
.left-panel {
    border-right: 1px solid var(--wire);
    padding: 2rem 1.75rem;
    display: flex; flex-direction: column; gap: 1.5rem;
    background: var(--ink-2);
}
.right-panel {
    padding: 2rem 2.5rem;
    overflow-y: auto;
}

/* ═══════════════════════════════════════════
   SECTION LABELS
══════════════════════════════════════════════ */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: var(--smoke);
    margin-bottom: 0.75rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--wire);
}

/* ═══════════════════════════════════════════
   DRAG & DROP ZONE
══════════════════════════════════════════════ */
.drop-zone {
    border: 1.5px dashed var(--wire-2);
    border-radius: 12px;
    padding: 2rem 1.25rem;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    background: var(--ink-3);
    position: relative;
}
.drop-zone:hover, .drop-zone.dragover {
    border-color: var(--lime);
    background: var(--lime-dim);
}
.drop-icon {
    width: 44px; height: 44px;
    border-radius: 10px;
    background: var(--ink-4);
    border: 1px solid var(--wire-2);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    margin: 0 auto 0.75rem;
}
.drop-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--white);
    margin-bottom: 0.3rem;
}
.drop-sub {
    font-size: 0.75rem;
    color: var(--smoke);
    line-height: 1.5;
}
.drop-formats {
    display: flex; gap: 0.4rem; flex-wrap: wrap; justify-content: center;
    margin-top: 0.75rem;
}
.fmt-badge {
    padding: 0.15rem 0.45rem;
    border-radius: 3px;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    font-weight: 700;
    background: var(--ink-4);
    color: var(--wire-3);
    border: 1px solid var(--wire);
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
#file-input {
    position: absolute; inset: 0; opacity: 0; cursor: pointer; width: 100%; height: 100%;
}

/* ═══════════════════════════════════════════
   OR DIVIDER
══════════════════════════════════════════════ */
.or-divider {
    display: flex; align-items: center; gap: 0.75rem;
    font-size: 0.7rem; color: var(--wire-3);
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.1em;
}
.or-divider::before, .or-divider::after {
    content: ''; flex: 1; height: 1px; background: var(--wire);
}

/* ═══════════════════════════════════════════
   PIPELINE STEPS (left panel)
══════════════════════════════════════════════ */
.pipeline-list { display: flex; flex-direction: column; gap: 0; }
.pipe-row {
    display: flex; align-items: flex-start; gap: 0.85rem;
    padding: 0.6rem 0;
    position: relative;
}
.pipe-row:not(:last-child)::after {
    content: '';
    position: absolute;
    left: 10px; top: 32px;
    width: 1px; height: calc(100% - 14px);
    background: var(--wire);
}
.pipe-node {
    width: 21px; height: 21px; flex-shrink: 0;
    border-radius: 50%;
    border: 1.5px solid var(--wire-2);
    background: var(--ink-3);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.6rem;
    margin-top: 1px;
    transition: all 0.3s;
}
.pipe-node.done   { background: var(--lime); border-color: var(--lime); color: var(--ink); font-weight: 700; }
.pipe-node.active { background: var(--ink-3); border-color: var(--lime); box-shadow: 0 0 8px var(--lime-glow); animation: blink 1.2s infinite; }
.pipe-text { flex: 1; }
.pipe-name {
    font-size: 0.82rem; font-weight: 500; color: var(--ash);
    transition: color 0.3s;
}
.pipe-name.done   { color: var(--white); }
.pipe-name.active { color: var(--lime); }
.pipe-detail {
    font-size: 0.7rem; color: var(--wire-3);
    font-family: 'Space Mono', monospace;
    margin-top: 0.1rem;
}

/* ═══════════════════════════════════════════
   INPUTS & FORM ELEMENTS
══════════════════════════════════════════════ */
.stTextInput > div > div > input {
    background: var(--ink-3) !important;
    border: 1px solid var(--wire-2) !important;
    border-radius: 8px !important;
    color: var(--white) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.6rem 0.9rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--lime) !important;
    box-shadow: 0 0 0 3px var(--lime-dim) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: var(--wire-3) !important; }

.stSelectbox > div > div {
    background: var(--ink-3) !important;
    border: 1px solid var(--wire-2) !important;
    border-radius: 8px !important;
    color: var(--white) !important;
    font-size: 0.82rem !important;
}

/* ═══════════════════════════════════════════
   BUTTONS
══════════════════════════════════════════════ */
.stButton > button {
    background: var(--lime) !important;
    color: var(--ink) !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.15s, transform 0.1s !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }
.stButton > button[kind="secondary"] {
    background: var(--ink-3) !important;
    color: var(--smoke) !important;
    border: 1px solid var(--wire-2) !important;
}

label { color: var(--smoke) !important; font-size: 0.75rem !important; letter-spacing: 0.04em !important; }
hr { border: none !important; border-top: 1px solid var(--wire) !important; margin: 0 !important; }

/* ═══════════════════════════════════════════
   RIGHT PANEL — RESULTS
══════════════════════════════════════════════ */
.result-title-bar {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 1.75rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--wire);
}
.result-title-text {
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: var(--white);
    line-height: 1.3;
    max-width: 70%;
}
.result-badge {
    padding: 0.3rem 0.75rem;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: var(--lime-dim);
    color: var(--lime);
    border: 1px solid rgba(200,241,53,0.25);
}

/* Stat chips row */
.chip-row { display: flex; gap: 0.75rem; flex-wrap: wrap; margin-bottom: 2rem; }
.stat-chip {
    background: var(--ink-3);
    border: 1px solid var(--wire);
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    display: flex; flex-direction: column; gap: 0.1rem;
}
.chip-val { font-size: 1.1rem; font-weight: 700; color: var(--white); letter-spacing: -0.02em; }
.chip-key { font-size: 0.65rem; color: var(--smoke); letter-spacing: 0.06em; text-transform: uppercase; font-family: 'Space Mono', monospace; }

/* Content grid */
.content-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.info-card {
    background: var(--ink-2);
    border: 1px solid var(--wire);
    border-radius: 10px;
    padding: 1.1rem 1.2rem;
}
.info-card-header {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 0.75rem;
}
.info-card-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--smoke);
}
.info-card-icon { font-size: 0.9rem; }
.info-card-body {
    font-size: 0.82rem; line-height: 1.75; color: var(--ash);
}

/* Summary card */
.summary-card {
    background: var(--ink-2);
    border: 1px solid var(--wire);
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.summary-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--lime);
    border-radius: 3px 0 0 3px;
}
.summary-card-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--lime); margin-bottom: 0.75rem;
}
.summary-card-body {
    font-size: 0.875rem; line-height: 1.8; color: var(--paper);
}

/* Transcript */
.transcript-card {
    background: var(--ink-3);
    border: 1px solid var(--wire);
    border-radius: 10px;
    padding: 1.25rem;
    max-height: 260px;
    overflow-y: auto;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.9;
    color: var(--smoke);
    white-space: pre-wrap;
    word-break: break-word;
    margin-bottom: 1.5rem;
}

/* ═══════════════════════════════════════════
   CHAT SECTION
══════════════════════════════════════════════ */
.chat-header {
    display: flex; align-items: center; gap: 0.6rem;
    margin-bottom: 1rem;
}
.chat-title {
    font-size: 0.95rem; font-weight: 700;
    letter-spacing: -0.01em; color: var(--white);
}
.chat-pill {
    font-size: 0.62rem; font-weight: 700;
    padding: 0.15rem 0.5rem;
    border-radius: 100px;
    background: var(--ink-4);
    color: var(--smoke);
    border: 1px solid var(--wire);
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.chat-box {
    background: var(--ink-2);
    border: 1px solid var(--wire);
    border-radius: 10px;
    padding: 1rem;
    max-height: 360px;
    overflow-y: auto;
    margin-bottom: 0.75rem;
    display: flex; flex-direction: column; gap: 1rem;
}
.chat-msg { display: flex; flex-direction: column; }
.chat-sender {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem; font-weight: 700;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.sender-user { color: var(--lime); padding-left: 1px; }
.sender-ai   { color: var(--blue); padding-left: 1px; }
.bubble {
    font-size: 0.84rem; line-height: 1.65;
    padding: 0.65rem 0.9rem;
    border-radius: 8px;
    max-width: 85%;
}
.bubble-user {
    background: var(--lime-dim);
    border: 1px solid rgba(200,241,53,0.15);
    color: var(--white);
    align-self: flex-end;
    border-radius: 8px 8px 2px 8px;
}
.bubble-ai {
    background: var(--blue-dim);
    border: 1px solid rgba(77,159,255,0.15);
    color: var(--white);
    align-self: flex-start;
    border-radius: 8px 8px 8px 2px;
}
.chat-empty {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 2.5rem; text-align: center;
    color: var(--wire-3);
    font-size: 0.82rem; line-height: 1.7;
}
.chat-empty-icon { font-size: 1.5rem; margin-bottom: 0.6rem; opacity: 0.4; }

/* ═══════════════════════════════════════════
   EMPTY / READY STATE
══════════════════════════════════════════════ */
.ready-state {
    display: flex; flex-direction: column;
    align-items: flex-start; justify-content: center;
    min-height: calc(100vh - 120px);
    padding: 3rem 0;
}
.ready-tag {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--lime);
    margin-bottom: 1.25rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.ready-h {
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 700;
    letter-spacing: -0.04em;
    line-height: 1.1;
    color: var(--white);
    margin-bottom: 1.1rem;
}
.ready-h span { color: var(--smoke); }
.ready-body {
    font-size: 0.92rem; line-height: 1.75;
    color: var(--ash); max-width: 460px;
    margin-bottom: 2.5rem;
}
.feature-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 0.75rem; width: 100%; max-width: 480px;
}
.feature-item {
    background: var(--ink-2);
    border: 1px solid var(--wire);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    display: flex; align-items: flex-start; gap: 0.65rem;
}
.feat-icon {
    width: 28px; height: 28px; flex-shrink: 0;
    border-radius: 7px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.85rem;
}
.feat-icon.lime   { background: var(--lime-dim);  }
.feat-icon.blue   { background: var(--blue-dim);  }
.feat-icon.amber  { background: var(--amber-dim); }
.feat-icon.red    { background: var(--red-dim);   }
.feat-name { font-size: 0.8rem; font-weight: 600; color: var(--white); }
.feat-desc { font-size: 0.72rem; color: var(--smoke); margin-top: 0.1rem; }

/* ═══════════════════════════════════════════
   MISC
══════════════════════════════════════════════ */
.stProgress > div > div > div { background: var(--lime) !important; }
.stSpinner > div { border-top-color: var(--lime) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--ash) !important; font-size: 0.875rem !important; }
.stExpander { background: var(--ink-2) !important; border: 1px solid var(--wire) !important; border-radius: 10px !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--wire-2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--lime); }

/* uploaded file notice */
.file-notice {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.55rem 0.8rem;
    border-radius: 8px;
    background: var(--lime-dim);
    border: 1px solid rgba(200,241,53,0.2);
    font-size: 0.78rem; color: var(--lime);
    font-family: 'Space Mono', monospace;
    margin-top: 0.5rem;
    word-break: break-all;
}
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
    "dropped_file_path": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── TOP NAV ────────────────────────────────────────────────────────────────────
status_label = "Ready" if not st.session_state.pipeline_done else "Analysis complete"
st.markdown(f"""
<div class="topbar">
    <div class="topbar-brand">
        <div class="brand-mark">▸</div>
        Video Aid
    </div>
    <div class="topbar-nav">
        <span class="nav-tag active">Analyse</span>
        <span class="nav-tag">History</span>
        <span class="nav-tag">Settings</span>
    </div>
    <div class="status-live">
        <div class="live-dot"></div>
        {status_label}
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Layout ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="layout">', unsafe_allow_html=True)
left_col, right_col = st.columns([340, 9999], gap="small")

# ════════════════════════════════════════════════════════
#  LEFT PANEL
# ════════════════════════════════════════════════════════
with left_col:
    st.markdown('<div class="left-panel">', unsafe_allow_html=True)

    # ── Drag & Drop zone ──────────────────────────────────
    st.markdown('<div class="section-label">Upload</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "drop_zone",
        type=["mp4", "mov", "avi", "mkv", "mp3", "wav", "m4a", "webm"],
        label_visibility="collapsed",
        key="file_uploader",
    )

    st.markdown("""
    <style>
    [data-testid="stFileUploader"] {
        background: var(--ink-3) !important;
        border: 1.5px dashed var(--wire-2) !important;
        border-radius: 12px !important;
        padding: 0 !important;
        transition: border-color 0.2s, background 0.2s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--lime) !important;
        background: var(--lime-dim) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--smoke) !important;
        font-size: 0.8rem !important;
    }
    [data-testid="stFileUploaderDropzone"] {
        background: transparent !important;
        padding: 1.75rem 1rem !important;
        text-align: center !important;
    }
    [data-testid="stFileUploaderDropzone"] svg { color: var(--wire-3) !important; }
    [data-testid="stFileUploaderDropzone"] small { color: var(--wire-3) !important; font-size: 0.7rem !important; }
    [data-testid="stFileUploaderDropzone"] button {
        background: var(--ink-4) !important;
        border: 1px solid var(--wire-2) !important;
        color: var(--ash) !important;
        border-radius: 6px !important;
        font-size: 0.78rem !important;
        padding: 0.3rem 0.75rem !important;
        font-family: 'Space Mono', monospace !important;
    }
    </style>
    """, unsafe_allow_html=True)

    if uploaded:
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1])
        tmp.write(uploaded.read()); tmp.flush()
        st.session_state.dropped_file_path = tmp.name
        st.markdown(f'<div class="file-notice">📎 {uploaded.name}</div>', unsafe_allow_html=True)

    # ── OR YouTube ────────────────────────────────────────
    st.markdown('<div class="or-divider">or</div>', unsafe_allow_html=True)
    yt_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=…", label_visibility="visible")

    # Resolve source
    source_val = st.session_state.dropped_file_path if uploaded else yt_url.strip()

    # ── Language ─────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:0.5rem">Language</div>', unsafe_allow_html=True)
    language = st.selectbox("Transcription language", ["english", "hinglish"], label_visibility="collapsed")

    # ── Run button ────────────────────────────────────────
    run_btn = st.button("▸  Run Analysis", use_container_width=True)

    # ── Pipeline status ───────────────────────────────────
    if st.session_state.pipeline_done or any(st.session_state.pipeline_steps.values()):
        st.markdown('<div class="section-label" style="margin-top:0.5rem">Pipeline</div>', unsafe_allow_html=True)

        steps_meta = [
            ("audio",      "Audio extraction",   "01"),
            ("transcript", "Transcription",       "02"),
            ("title",      "Title generation",    "03"),
            ("summary",    "Summarisation",       "04"),
            ("extract",    "Data extraction",     "05"),
            ("rag",        "RAG engine build",    "06"),
        ]
        st.markdown('<div class="pipeline-list">', unsafe_allow_html=True)
        for key, label, num in steps_meta:
            s = st.session_state.pipeline_steps.get(key, "pending")
            node_cls = s  # "pending"|"active"|"done"
            inner = "✓" if s == "done" else ("" if s == "active" else num)
            name_cls = "done" if s == "done" else ("active" if s == "active" else "")
            st.markdown(f"""
            <div class="pipe-row">
                <div class="pipe-node {node_cls}">{inner}</div>
                <div class="pipe-text">
                    <div class="pipe-name {name_cls}">{label}</div>
                    <div class="pipe-detail">{num} / 06</div>
                </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # left-panel

# ════════════════════════════════════════════════════════
#  RIGHT PANEL
# ════════════════════════════════════════════════════════
with right_col:
    st.markdown('<div class="right-panel">', unsafe_allow_html=True)

    # ── Pipeline run ────────────────────────────────────────────────────────────
    if run_btn:
        if not source_val:
            st.error("Provide a YouTube URL or drop a file first.")
        else:
            st.session_state.pipeline_done = False
            st.session_state.result = None
            st.session_state.chat_history = []
            st.session_state.pipeline_steps = {}

            ph = st.empty()

            def set_step(k, v):
                st.session_state.pipeline_steps[k] = v

            try:
                ph.info("Pipeline initialised — processing your media…")

                set_step("audio", "active");      chunks     = process_input(source_val);                     set_step("audio", "done")
                set_step("transcript", "active"); transcript = transcribe_all(chunks, language);              set_step("transcript", "done")
                set_step("title", "active");      title      = generate_title(transcript);                   set_step("title", "done")
                set_step("summary", "active");    summary    = summarize(transcript);                         set_step("summary", "done")
                set_step("extract", "active");    action_items = extract_action_items(transcript); decisions = extract_key_decisions(transcript); questions = extract_questions(transcript); set_step("extract", "done")
                set_step("rag", "active");        rag_chain  = build_rag_chain(transcript);                  set_step("rag", "done")

                st.session_state.result = {
                    "title": title, "transcript": transcript,
                    "summary": summary, "action_items": action_items,
                    "key_decisions": decisions, "open_questions": questions,
                    "rag_chain": rag_chain,
                }
                st.session_state.pipeline_done = True
                ph.success("Analysis complete.")
                time.sleep(0.4); ph.empty(); st.rerun()

            except Exception as e:
                for k in ["audio","transcript","title","summary","extract","rag"]:
                    if st.session_state.pipeline_steps.get(k) == "active":
                        st.session_state.pipeline_steps[k] = "pending"
                ph.error(f"Pipeline error: {e}")

    # ── Results ─────────────────────────────────────────────────────────────────
    if st.session_state.result:
        r = st.session_state.result

        # Title bar
        word_count = len(r["transcript"].split()) if r["transcript"] else 0
        st.markdown(f"""
        <div class="result-title-bar">
            <div class="result-title-text">{r['title']}</div>
            <div class="result-badge">✓ Complete</div>
        </div>
        <div class="chip-row">
            <div class="stat-chip"><div class="chip-val">{word_count:,}</div><div class="chip-key">Words</div></div>
            <div class="stat-chip"><div class="chip-val">{len(r['transcript'].split(chr(10)))}</div><div class="chip-key">Lines</div></div>
            <div class="stat-chip"><div class="chip-val">6/6</div><div class="chip-key">Steps done</div></div>
            <div class="stat-chip"><div class="chip-val">RAG</div><div class="chip-key">Chat ready</div></div>
        </div>
        """, unsafe_allow_html=True)

        # Summary
        st.markdown(f"""
        <div class="summary-card">
            <div class="summary-card-label">▸ Summary</div>
            <div class="summary-card-body">{r['summary']}</div>
        </div>
        """, unsafe_allow_html=True)

        # 3-col cards
        st.markdown(f"""
        <div class="content-grid">
            <div class="info-card">
                <div class="info-card-header">
                    <div class="info-card-label">Action Items</div>
                    <div class="info-card-icon">✅</div>
                </div>
                <div class="info-card-body">{r['action_items']}</div>
            </div>
            <div class="info-card">
                <div class="info-card-header">
                    <div class="info-card-label">Key Decisions</div>
                    <div class="info-card-icon">🔑</div>
                </div>
                <div class="info-card-body">{r['key_decisions']}</div>
            </div>
            <div class="info-card">
                <div class="info-card-header">
                    <div class="info-card-label">Open Questions</div>
                    <div class="info-card-icon">❓</div>
                </div>
                <div class="info-card-body">{r['open_questions']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Transcript (collapsible)
        with st.expander("▸ Full Transcript", expanded=False):
            st.markdown(f'<div class="transcript-card">{r["transcript"]}</div>', unsafe_allow_html=True)

        st.markdown("<hr style='margin:1.5rem 0'>", unsafe_allow_html=True)

        # Chat
        n = len([m for m in st.session_state.chat_history if m["role"] == "user"])
        st.markdown(f"""
        <div class="chat-header">
            <div class="chat-title">Ask the transcript</div>
            <div class="chat-pill">{n} message{"s" if n != 1 else ""}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.chat_history:
            rows = ""
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    rows += f"""
                    <div class="chat-msg">
                        <span class="chat-sender sender-user">You</span>
                        <div class="bubble bubble-user">{msg['content']}</div>
                    </div>"""
                else:
                    rows += f"""
                    <div class="chat-msg">
                        <span class="chat-sender sender-ai">Video Aid</span>
                        <div class="bubble bubble-ai">{msg['content']}</div>
                    </div>"""
            st.markdown(f'<div class="chat-box">{rows}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="chat-box">
                <div class="chat-empty">
                    <div class="chat-empty-icon">◈</div>
                    Ask anything about the video — key topics, who said what, next steps, timelines…
                </div>
            </div>
            """, unsafe_allow_html=True)

        ci1, ci2 = st.columns([6, 1], gap="small")
        with ci1:
            user_q = st.text_input("q", placeholder="e.g. What decisions were made?", label_visibility="collapsed")
        with ci2:
            send = st.button("Send", use_container_width=True)

        if send and user_q.strip():
            with st.spinner(""):
                ans = ask_question(r["rag_chain"], user_q.strip())
            st.session_state.chat_history.append({"role": "user",      "content": user_q.strip()})
            st.session_state.chat_history.append({"role": "assistant", "content": ans})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

    else:
        # ── Empty / ready state ──────────────────────────────────────────────────
        st.markdown("""
        <div class="ready-state">
            <div class="ready-tag"><div class="live-dot"></div> Video Aid — v2</div>
            <h1 class="ready-heading" style="font-size:clamp(2rem,4vw,3.2rem);font-weight:700;letter-spacing:-0.04em;line-height:1.1;color:var(--white);margin-bottom:1.1rem">
                Transcribe. Extract.<br><span style="color:var(--smoke)">Understand anything.</span>
            </h1>
            <p class="ready-body">
                Drop a video or audio file on the left — or paste a YouTube link — and Video Aid will
                transcribe it, generate a summary, extract action items and decisions, then open a
                live chat so you can ask questions directly against the content.
            </p>
            <div class="feature-grid">
                <div class="feature-item">
                    <div class="feat-icon lime">📝</div>
                    <div><div class="feat-name">Transcription</div><div class="feat-desc">Accurate speech-to-text with speaker lines</div></div>
                </div>
                <div class="feature-item">
                    <div class="feat-icon blue">📋</div>
                    <div><div class="feat-name">Summarisation</div><div class="feat-desc">Concise AI-generated meeting summary</div></div>
                </div>
                <div class="feature-item">
                    <div class="feat-icon amber">🔍</div>
                    <div><div class="feat-name">Extraction</div><div class="feat-desc">Action items, decisions & open questions</div></div>
                </div>
                <div class="feature-item">
                    <div class="feat-icon red">🧠</div>
                    <div><div class="feat-name">RAG Chat</div><div class="feat-desc">Ask anything, get grounded answers</div></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # right-panel

st.markdown('</div>', unsafe_allow_html=True)  # layout
