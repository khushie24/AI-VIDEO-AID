import streamlit as st
import time
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MeetMind — AI Video Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

:root {
    --bg:          #0c0c0e;
    --surface-1:   #111115;
    --surface-2:   #18181f;
    --surface-3:   #1e1e28;
    --border-1:    rgba(255,255,255,0.06);
    --border-2:    rgba(255,255,255,0.10);
    --border-3:    rgba(255,255,255,0.16);
    --violet:      #8b5cf6;
    --violet-dim:  rgba(139,92,246,0.12);
    --violet-mid:  rgba(139,92,246,0.25);
    --emerald:     #10b981;
    --emerald-dim: rgba(16,185,129,0.10);
    --sky:         #38bdf8;
    --sky-dim:     rgba(56,189,248,0.10);
    --amber:       #f59e0b;
    --amber-dim:   rgba(245,158,11,0.10);
    --red:         #f87171;
    --text-1:      #f0f0f5;
    --text-2:      #a0a0b8;
    --text-3:      #606078;
}

html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
    background-color: var(--bg) !important;
    color: var(--text-1) !important;
}

.stApp { background: var(--bg) !important; }

/* Subtle radial glow behind content */
.stApp::after {
    content: '';
    position: fixed;
    top: -20vh; right: -20vw;
    width: 60vw; height: 60vh;
    background: radial-gradient(ellipse, rgba(139,92,246,0.06) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface-1) !important;
    border-right: 1px solid var(--border-1) !important;
}
[data-testid="stSidebar"] * { color: var(--text-1) !important; }
[data-testid="stSidebar"] .block-container { padding: 2rem 1.25rem !important; }

/* ── Typography ── */
h1, h2, h3, h4 {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-1) !important;
}

/* ── Wordmark ── */
.wordmark {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.25rem;
}
.wordmark-icon {
    width: 32px; height: 32px;
    background: var(--violet-mid);
    border: 1px solid rgba(139,92,246,0.35);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; line-height: 1;
    color: var(--violet);
}
.wordmark-text {
    font-family: 'Inter', sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: var(--text-1);
}
.wordmark-sub {
    font-size: 0.72rem;
    color: var(--text-3);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-left: 0.2rem;
    margin-bottom: 1.5rem;
}

/* ── Hero ── */
.hero {
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--violet);
    margin-bottom: 0.5rem;
}
.hero-heading {
    font-family: 'Inter', sans-serif;
    font-size: clamp(1.8rem, 4vw, 2.75rem);
    font-weight: 600;
    letter-spacing: -0.04em;
    line-height: 1.15;
    color: var(--text-1);
    margin: 0 0 0.6rem;
}
.hero-heading em {
    font-style: normal;
    color: var(--violet);
}
.hero-desc {
    font-size: 0.9rem;
    color: var(--text-2);
    line-height: 1.65;
    max-width: 520px;
}

/* ── Pill Tags ── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    border: 1px solid;
}
.pill-violet { background: var(--violet-dim);  color: #c4b5fd; border-color: rgba(139,92,246,0.2); }
.pill-sky    { background: var(--sky-dim);     color: #7dd3fc; border-color: rgba(56,189,248,0.2); }
.pill-emerald{ background: var(--emerald-dim); color: #6ee7b7; border-color: rgba(16,185,129,0.2); }
.pill-amber  { background: var(--amber-dim);   color: #fcd34d; border-color: rgba(245,158,11,0.2); }

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid var(--border-1) !important;
    margin: 1.5rem 0 !important;
}

/* ── Cards ── */
.card {
    background: var(--surface-1);
    border: 1px solid var(--border-1);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
    position: relative;
}
.card-sm {
    background: var(--surface-1);
    border: 1px solid var(--border-1);
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
    position: relative;
}
.card-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.card-label-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--violet);
    flex-shrink: 0;
}
.card-body {
    font-size: 0.875rem;
    line-height: 1.75;
    color: var(--text-2);
}

/* Title card special style */
.title-card {
    background: linear-gradient(135deg, var(--surface-2) 0%, var(--surface-1) 100%);
    border: 1px solid var(--border-2);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.title-card-icon {
    width: 38px; height: 38px; flex-shrink: 0;
    background: var(--violet-dim);
    border: 1px solid rgba(139,92,246,0.25);
    border-radius: 9px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.title-card-text {
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text-1);
    line-height: 1.3;
}
.title-card-sub {
    font-size: 0.72rem;
    color: var(--text-3);
    margin-top: 0.2rem;
    letter-spacing: 0.04em;
}

/* ── Status Steps (sidebar) ── */
.step-item {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    padding: 0.5rem 0.75rem;
    border-radius: 8px;
    margin-bottom: 0.25rem;
    font-size: 0.8rem;
    color: var(--text-2);
    transition: background 0.2s;
}
.step-item.done   { color: var(--text-1); background: rgba(16,185,129,0.06); }
.step-item.active { color: var(--text-1); background: var(--violet-dim); }
.step-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.dot-pending { background: var(--border-2); }
.dot-active  { background: var(--violet);
                box-shadow: 0 0 6px var(--violet);
                animation: blink 1.4s ease-in-out infinite; }
.dot-done    { background: var(--emerald); }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.35} }

/* ── Inputs ── */
.stTextInput > div > div > input {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: 8px !important;
    color: var(--text-1) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.875rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--violet) !important;
    box-shadow: 0 0 0 3px var(--violet-dim) !important;
}
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-2) !important;
    border-radius: 8px !important;
    color: var(--text-1) !important;
    font-size: 0.875rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: var(--violet) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.55rem 1.25rem !important;
    transition: opacity 0.15s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.stButton > button[kind="secondary"] {
    background: var(--surface-3) !important;
    color: var(--text-2) !important;
    border: 1px solid var(--border-2) !important;
}

/* ── Transcript Expander ── */
.stExpander {
    background: var(--surface-1) !important;
    border: 1px solid var(--border-1) !important;
    border-radius: 10px !important;
}
.stExpander summary { color: var(--text-2) !important; font-size: 0.85rem !important; }

/* ── Transcript Box ── */
.transcript-box {
    background: var(--surface-2);
    border-radius: 8px;
    padding: 1rem 1.1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.85;
    color: var(--text-2);
    max-height: 320px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-word;
}

/* ── Chat ── */
.chat-wrap {
    background: var(--surface-1);
    border: 1px solid var(--border-1);
    border-radius: 12px;
    padding: 1.25rem;
    max-height: 400px;
    overflow-y: auto;
    margin-bottom: 1rem;
    display: flex;
    flex-direction: column;
    gap: 1.1rem;
}
.chat-row { display: flex; flex-direction: column; gap: 0.2rem; }
.chat-who {
    font-size: 0.64rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.chat-who.user { color: var(--violet); padding-left: 2px; }
.chat-who.bot  { color: var(--sky);    padding-left: 2px; }
.chat-bubble {
    padding: 0.65rem 1rem;
    border-radius: 10px;
    font-size: 0.86rem;
    line-height: 1.65;
    max-width: 88%;
}
.user-bubble {
    background: var(--violet-dim);
    border: 1px solid rgba(139,92,246,0.18);
    align-self: flex-end;
    color: var(--text-1);
}
.bot-bubble {
    background: var(--sky-dim);
    border: 1px solid rgba(56,189,248,0.15);
    align-self: flex-start;
    color: var(--text-1);
}

/* ── Empty State ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 5rem 2rem;
    text-align: center;
}
.empty-icon {
    width: 56px; height: 56px;
    background: var(--surface-2);
    border: 1px solid var(--border-2);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.6rem;
    margin-bottom: 1.25rem;
}
.empty-heading {
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: -0.02em;
    color: var(--text-1);
    margin-bottom: 0.45rem;
}
.empty-body {
    font-size: 0.85rem;
    color: var(--text-2);
    line-height: 1.7;
    max-width: 360px;
    margin-bottom: 1.5rem;
}

/* ── Misc ── */
.stProgress > div > div > div { background: var(--violet) !important; }
.stSpinner > div { border-top-color: var(--violet) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text-2) !important; }
label { color: var(--text-3) !important; font-size: 0.78rem !important; letter-spacing: 0.02em !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-2); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--violet); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "processing": False,
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
def step_css(key: str) -> str:
    s = st.session_state.pipeline_steps.get(key, "pending")
    return s  # "pending" | "active" | "done"

def render_step(label: str, key: str, icon: str):
    state = step_css(key)
    dot   = f"dot-{state}"
    cls   = f"step-item {state}"
    st.markdown(f"""
    <div class="{cls}">
        <div class="step-dot {dot}"></div>
        <span>{icon}&nbsp; {label}</span>
    </div>""", unsafe_allow_html=True)

# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="wordmark">
        <div class="wordmark-icon">◈</div>
        <span class="wordmark-text">MeetMind</span>
    </div>
    <div class="wordmark-sub">AI Video Intelligence</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="pill pill-violet" style="margin-bottom:0.75rem">⚙ Input</div>', unsafe_allow_html=True)
    source = st.text_input("YouTube URL or File Path", placeholder="https://youtube.com/watch?v=... or /path/to/file")
    language = st.selectbox("Transcription Language", ["english", "hinglish"], index=0)
    run_btn = st.button("▶  Run Analysis", use_container_width=True)

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<div class="pill pill-emerald" style="margin-bottom:0.75rem">● Pipeline Status</div>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio",      "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷", "Title Generation"),
            ("summary",    "📋", "Summarisation"),
            ("extract",    "🔍", "Extraction"),
            ("rag",        "🧠", "RAG Engine"),
        ]:
            render_step(label, step, icon)

# ─── Main ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">◈ MeetMind</div>
    <h1 class="hero-heading">Meeting Intelligence<br><em>at your fingertips</em></h1>
    <p class="hero-desc">Paste a YouTube link or file path, pick a language, and get a full transcript, summary, action items, decisions, and an AI chat interface — in seconds.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── Run Pipeline ────────────────────────────────────────────────────────────────
if run_btn:
    if not source.strip():
        st.error("Please provide a YouTube URL or file path.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        try:
            placeholder.info("⚙️ Pipeline running — watch the sidebar for live progress.")

            update_step("audio", "active")
            chunks = process_input(source)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language)
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions    = extract_key_decisions(transcript)
            questions    = extract_questions(transcript)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title":          title,
                "transcript":     transcript,
                "summary":        summary,
                "action_items":   action_items,
                "key_decisions":  decisions,
                "open_questions": questions,
                "rag_chain":      rag_chain,
            }
            st.session_state.pipeline_done = True
            placeholder.success("✅ Analysis complete!")
            time.sleep(0.5)
            placeholder.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio","transcript","title","summary","extract","rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            placeholder.error(f"❌ Pipeline error: {e}")

# ─── Results ─────────────────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Session title
    st.markdown(f"""
    <div class="title-card">
        <div class="title-card-icon">📌</div>
        <div>
            <div class="title-card-text">{r['title']}</div>
            <div class="title-card-sub">Session title generated by AI</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Summary + Transcript
    col1, col2 = st.columns([3, 2], gap="medium")
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-label"><div class="card-label-dot"></div>Summary</div>
            <div class="card-body">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("Full Transcript", expanded=False):
            st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)

    # Action items / Decisions / Questions
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(f"""
        <div class="card-sm">
            <div class="card-label"><div class="card-label-dot" style="background:var(--emerald)"></div>Action Items</div>
            <div class="card-body">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card-sm">
            <div class="card-label"><div class="card-label-dot" style="background:var(--amber)"></div>Key Decisions</div>
            <div class="card-body">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card-sm">
            <div class="card-label"><div class="card-label-dot" style="background:var(--sky)"></div>Open Questions</div>
            <div class="card-body">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Chat
    st.markdown('<div style="font-size:0.95rem;font-weight:600;letter-spacing:-0.01em;margin-bottom:0.85rem;color:var(--text-1)">💬 Chat with your Meeting</div>', unsafe_allow_html=True)

    if st.session_state.chat_history:
        rows = ""
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                rows += f"""
                <div class="chat-row" style="align-items:flex-end">
                    <span class="chat-who user">You</span>
                    <div class="chat-bubble user-bubble">{msg['content']}</div>
                </div>"""
            else:
                rows += f"""
                <div class="chat-row" style="align-items:flex-start">
                    <span class="chat-who bot">Assistant</span>
                    <div class="chat-bubble bot-bubble">{msg['content']}</div>
                </div>"""
        st.markdown(f'<div class="chat-wrap">{rows}</div>', unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem 1.5rem">
            <div style="font-size:1.5rem;margin-bottom:0.6rem;opacity:0.5">💬</div>
            <div style="font-size:0.85rem;color:var(--text-3)">Ask anything about your meeting — decisions, next steps, participants, topics…</div>
        </div>""", unsafe_allow_html=True)

    chat_col1, chat_col2 = st.columns([5, 1], gap="small")
    with chat_col1:
        user_input = st.text_input("question", placeholder="e.g. What were the main decisions made?", label_visibility="collapsed")
    with chat_col2:
        send_btn = st.button("Send", use_container_width=True)

    if send_btn and user_input.strip():
        with st.spinner("Thinking…"):
            answer = ask_question(r["rag_chain"], user_input.strip())
        st.session_state.chat_history.append({"role": "user",      "content": user_input.strip()})
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

    if st.session_state.chat_history:
        if st.button("Clear chat", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

else:
    # Empty state
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">◈</div>
        <div class="empty-heading">Nothing here yet</div>
        <div class="empty-body">Paste a YouTube URL or local file path in the sidebar, select a transcription language, and hit <strong style="color:var(--text-1)">Run Analysis</strong> to begin.</div>
        <div style="display:flex;gap:0.6rem;flex-wrap:wrap;justify-content:center">
            <span class="pill pill-violet">Transcription</span>
            <span class="pill pill-sky">Summarisation</span>
            <span class="pill pill-emerald">Action Items</span>
            <span class="pill pill-amber">RAG Chat</span>
        </div>
    </div>""", unsafe_allow_html=True)
