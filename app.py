import streamlit as st
import time
import tempfile
import os
from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Aid",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg:        #F5F4F0;
    --surface:   #FFFFFF;
    --border:    rgba(0,0,0,0.10);
    --border-md: rgba(0,0,0,0.18);
    --text:      #1A1A1A;
    --muted:     #6B6B6B;
    --accent:    #1C6EF2;
    --accent-bg: #EBF2FF;
    --green:     #0D7A55;
    --green-bg:  #E4F5EE;
    --amber:     #905A00;
    --amber-bg:  #FFF3D4;
    --radius:    10px;
    --radius-lg: 14px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: var(--bg) !important;
    color: var(--text) !important;
}
.stApp { background: var(--bg) !important; }

/* Hide default sidebar toggle */
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 14px 28px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    margin: -1rem -1rem 1.5rem -1rem;
}
.logo {
    display: flex;
    align-items: center;
    gap: 10px;
}
.logo-icon {
    width: 34px; height: 34px;
    border-radius: 8px;
    background: var(--accent-bg);
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
}
.logo-wordmark {
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.02em;
}
.logo-sub {
    font-size: 11px;
    color: var(--muted);
    margin-top: 1px;
}


.section-label {
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}

/* ── Divider ── */
.or-divider {
    display: flex; align-items: center; gap: 10px;
    font-size: 11px; color: var(--muted);
}
.or-divider::before, .or-divider::after {
    content: ''; flex: 1;
    height: 1px; background: var(--border);
}

/* ── Format tags ── */
.fmt-row { display: flex; gap: 5px; justify-content: center; margin-top: 10px; flex-wrap: wrap; }
.fmt-tag {
    font-size: 10px; font-weight: 500;
    padding: 2px 8px; border-radius: 20px;
    background: var(--accent-bg); color: var(--accent);
    border: 1px solid rgba(28,110,242,0.2);
}

/* ── Pipeline steps ── */
.step-row {
    display: flex; align-items: center; gap: 9px;
    padding: 7px 10px;
    border-radius: 8px;
    background: var(--bg);
    border: 1px solid var(--border);
    margin-bottom: 4px;
}
.sdot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
.sdot-done    { background: #0D7A55; }
.sdot-active  { background: var(--accent); }
.sdot-pending { background: #D0D0D0; }
.step-name { font-size: 12px; flex: 1; color: var(--text); }
.step-badge {
    font-size: 10px; font-weight: 500;
    padding: 2px 8px; border-radius: 20px;
}
.sb-done    { background: var(--green-bg);  color: var(--green); }
.sb-active  { background: var(--accent-bg); color: var(--accent); }
.sb-pending { background: #F0F0F0; color: var(--muted); }

/* ── Right panel cards ── */
.title-bar {
    display: flex; align-items: center; justify-content: space-between;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 14px 18px;
    margin-bottom: 14px;
}
.title-text { font-size: 14px; font-weight: 600; color: var(--text); }
.title-meta { font-size: 11px; color: var(--muted); }

.summary-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 16px 18px;
    margin-bottom: 14px;
}
.card-label {
    font-size: 10.5px; font-weight: 600;
    letter-spacing: 0.07em; text-transform: uppercase;
    color: var(--muted); margin-bottom: 8px;
    display: flex; align-items: center; gap: 6px;
}
.card-body { font-size: 13px; line-height: 1.75; color: var(--text); }

.three-col { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-bottom: 14px; }
.mini-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 14px 16px;
}

/* ── Chat ── */
.chat-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    overflow: hidden;
}
.chat-header {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    font-size: 12px; font-weight: 500; color: var(--muted);
    background: var(--bg);
    display: flex; align-items: center; gap: 7px;
}
.chat-history {
    padding: 14px 16px;
    max-height: 280px;
    overflow-y: auto;
    display: flex; flex-direction: column; gap: 10px;
}
.msg-user {
    align-self: flex-end;
    background: var(--accent-bg);
    color: var(--accent);
    padding: 7px 12px;
    border-radius: 10px 10px 2px 10px;
    font-size: 12.5px;
    max-width: 80%;
}
.msg-bot {
    align-self: flex-start;
    background: var(--bg);
    color: var(--text);
    border: 1px solid var(--border);
    padding: 7px 12px;
    border-radius: 10px 10px 10px 2px;
    font-size: 12.5px;
    max-width: 82%;
    line-height: 1.6;
}
.chat-empty {
    padding: 2.5rem 1rem;
    text-align: center;
    color: var(--muted);
    font-size: 12.5px;
}

/* ── Stale streamlit bits ── */
.stProgress > div > div > div { background: var(--accent) !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--text) !important; }
label { color: var(--muted) !important; font-size: 12px !important; }

.stTextInput > div > div > input {
    background: var(--bg) !important;
    border: 1px solid var(--border-md) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 13px !important;
}
.stTextInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(28,110,242,0.12) !important;
}
.stButton > button {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 0.55rem 1.4rem !important;
    transition: background 0.15s !important;
}
.stButton > button:hover { background: #1558C8 !important; }
.stButton > button[kind="secondary"] {
    background: var(--bg) !important;
    color: var(--muted) !important;
    border: 1px solid var(--border-md) !important;
}
div[data-testid="stFileUploader"] {
    background: var(--bg) !important;
    border: 1.5px dashed var(--border-md) !important;
    border-radius: var(--radius) !important;
}

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-md); border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ───────────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── Helpers ────────────────────────────────────────────────────────────────────
STEPS = [
    ("audio",      "Audio extraction"),
    ("transcript", "Transcription"),
    ("summary",    "Summarisation"),
    ("extract",    "Extraction"),
    ("rag",        "RAG engine"),
]

def badge(key):
    s = st.session_state.pipeline_steps.get(key, "pending")
    css = {"active": "sb-active", "done": "sb-done"}.get(s, "sb-pending")
    dot = {"active": "sdot-active", "done": "sdot-done"}.get(s, "sdot-pending")
    label = {"active": "Running", "done": "Done"}.get(s, "Waiting")
    return dot, css, label

def update_step(key, state):
    st.session_state.pipeline_steps[key] = state

# ─── Top bar ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div class="logo">
    <div class="logo-icon">📹</div>
    <div>
      <div class="logo-wordmark">Video Aid</div>
      <div class="logo-sub">AI video intelligence</div>
    </div>
  </div>
  <div style="font-size:12px;color:var(--muted)">Transcribe · Summarise · Chat</div>
</div>
""", unsafe_allow_html=True)

# ─── Main two-column layout ───────────────────────────────────────────────────────
left_col, right_col = st.columns([1.1, 2.2], gap="medium")

with left_col:
    # ── Drag & drop file upload ──
    st.markdown('<div class="section-label">Upload video file</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Drop your video here",
        type=["mp4", "mov", "avi", "mkv", "webm"],
        label_visibility="collapsed",
    )
    st.markdown("""
    <div class="fmt-row" style="margin-top:6px">
      <span class="fmt-tag">MP4</span><span class="fmt-tag">MOV</span>
      <span class="fmt-tag">AVI</span><span class="fmt-tag">MKV</span><span class="fmt-tag">WEBM</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="or-divider" style="margin:10px 0">or paste a link</div>', unsafe_allow_html=True)

    # ── YouTube URL ──
    st.markdown('<div class="section-label">YouTube URL</div>', unsafe_allow_html=True)
    yt_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")

    # ── Options ──
    opt1, opt2 = st.columns(2, gap="small")
    with opt1:
        st.markdown('<div class="section-label">Language</div>', unsafe_allow_html=True)
        language = st.selectbox("Language", ["English", "Hinglish"], label_visibility="collapsed")
    with opt2:
        st.markdown('<div class="section-label">Output</div>', unsafe_allow_html=True)
        output_mode = st.selectbox("Output", ["Full report", "Summary only"], label_visibility="collapsed")

    run_btn = st.button("▶ Run analysis", use_container_width=True)

    # ── Pipeline status ──
    if st.session_state.pipeline_steps:
        st.markdown('<div class="section-label" style="margin-top:8px;margin-bottom:6px">Pipeline</div>', unsafe_allow_html=True)
        for key, name in STEPS:
            dot, css, label = badge(key)
            st.markdown(f"""
            <div class="step-row">
              <div class="sdot {dot}"></div>
              <span class="step-name">{name}</span>
              <span class="step-badge {css}">{label}</span>
            </div>""", unsafe_allow_html=True)

# ─── Pipeline execution ──────────────────────────────────────────────────────────
if run_btn:
    source = uploaded_file or yt_url.strip()
    if not source:
        with left_col:
            st.error("Upload a file or enter a YouTube URL.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        status_ph = right_col.empty()

        try:
            status_ph.info("⚙️ Pipeline running…")

            update_step("audio", "active")
            if isinstance(source, str):
                source_path = source
            else:
                suffix = os.path.splitext(source.name)[-1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(source.read())
                tmp.flush()
                tmp.close()
                source_path = tmp.name
            chunks = process_input(source_path)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language.lower())
            update_step("transcript", "done")

            update_step("summary", "active")
            title   = generate_title(transcript)
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
                "title": title, "transcript": transcript,
                "summary": summary, "action_items": action_items,
                "key_decisions": decisions, "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            status_ph.success("✅ Analysis complete!")
            time.sleep(0.4)
            status_ph.empty()
            st.rerun()

        except Exception as e:
            for k, _ in STEPS:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            status_ph.error(f"❌ {e}")

# ─── Results ─────────────────────────────────────────────────────────────────────
with right_col:
    if st.session_state.result:
        r = st.session_state.result

        # Title bar
        st.markdown(f"""
        <div class="title-bar">
          <div class="title-text">📌 {r['title']}</div>
          <div class="title-meta">Analysis complete</div>
        </div>""", unsafe_allow_html=True)

        # Summary
        st.markdown(f"""
        <div class="summary-card">
          <div class="card-label">📋 Summary</div>
          <div class="card-body">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

        # Three columns
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            st.markdown(f"""
            <div class="mini-card">
              <div class="card-label">✅ Action items</div>
              <div class="card-body">{r['action_items']}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="mini-card">
              <div class="card-label">🔑 Key decisions</div>
              <div class="card-body">{r['key_decisions']}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="mini-card">
              <div class="card-label">❓ Open questions</div>
              <div class="card-body">{r['open_questions']}</div>
            </div>""", unsafe_allow_html=True)

        # Transcript expander
        with st.expander("📝 Full transcript"):
            st.text_area("", value=r["transcript"], height=220, label_visibility="collapsed")

        st.markdown("---")

        # Chat
        st.markdown("""
        <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:10px">
          💬 Chat with this video
        </div>""", unsafe_allow_html=True)

        if st.session_state.chat_history:
            chat_html = '<div class="chat-wrap"><div class="chat-header">💬 Conversation</div><div class="chat-history">'
            for msg in st.session_state.chat_history:
                cls = "msg-user" if msg["role"] == "user" else "msg-bot"
                chat_html += f'<div class="{cls}">{msg["content"]}</div>'
            chat_html += '</div></div>'
            st.markdown(chat_html, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="chat-wrap">
              <div class="chat-header">💬 Conversation</div>
              <div class="chat-empty">Ask anything about the video transcript</div>
            </div>""", unsafe_allow_html=True)

        chat_c1, chat_c2 = st.columns([5, 1], gap="small")
        with chat_c1:
            user_q = st.text_input("Ask a question", placeholder="What were the main decisions?", label_visibility="collapsed")
        with chat_c2:
            send = st.button("Send", use_container_width=True)

        if send and user_q.strip():
            with st.spinner("Thinking…"):
                answer = ask_question(r["rag_chain"], user_q.strip())
            st.session_state.chat_history.append({"role": "user", "content": user_q.strip()})
            st.session_state.chat_history.append({"role": "assistant", "content": answer})
            st.rerun()

        if st.session_state.chat_history:
            if st.button("Clear chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()

    else:
        # Empty state
        st.markdown("""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    padding:5rem 2rem;text-align:center;
                    background:var(--surface);border:1px solid var(--border);
                    border-radius:var(--radius-lg);">
          <div style="font-size:3rem;margin-bottom:16px">📹</div>
          <div style="font-size:16px;font-weight:600;color:var(--text);margin-bottom:8px">
            Ready when you are
          </div>
          <div style="font-size:13px;color:var(--muted);max-width:340px;line-height:1.7">
            Upload a local video file or paste a YouTube URL on the left, then hit <strong>Run analysis</strong>.
          </div>
          <div style="margin-top:20px;display:flex;gap:8px;flex-wrap:wrap;justify-content:center">
            <span style="font-size:11px;padding:4px 12px;border-radius:20px;
                         background:var(--accent-bg);color:var(--accent);
                         border:1px solid rgba(28,110,242,0.2)">Transcription</span>
            <span style="font-size:11px;padding:4px 12px;border-radius:20px;
                         background:var(--green-bg);color:var(--green);
                         border:1px solid rgba(13,122,85,0.2)">Summarisation</span>
            <span style="font-size:11px;padding:4px 12px;border-radius:20px;
                         background:var(--amber-bg);color:var(--amber);
                         border:1px solid rgba(144,90,0,0.2)">RAG Chat</span>
          </div>
        </div>""", unsafe_allow_html=True)
