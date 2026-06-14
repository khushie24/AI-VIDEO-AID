import streamlit as st
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarize import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions
)
from core.rag_engine import build_rag_chain, ask_question


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎥",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    max-width: 1400px;
}

.stButton > button {
    width: 100%;
    height: 3rem;
    border-radius: 12px;
    font-weight: bold;
    font-size: 16px;
}

[data-testid="stFileUploader"] {
    border: 2px dashed #4F46E5;
    border-radius: 15px;
    padding: 20px;
}

.chat-box {
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# PIPELINE
# -------------------------------------------------

def run_pipeline(source, language="english"):

    chunks = process_input(source)

    transcript = transcribe_all(
        chunks,
        language=language
    )

    title = generate_title(transcript)

    summary = summarize(transcript)

    action_items = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)

    questions = extract_questions(transcript)

    rag_chain = build_rag_chain(transcript)

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "decisions": decisions,
        "questions": questions,
        "rag_chain": rag_chain
    }

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------

if "result" not in st.session_state:
    st.session_state.result = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.title("🎥 AI Meeting & Video Assistant")

st.markdown(
    """
Upload a local meeting/video file or provide a YouTube URL.

The assistant will automatically:

- 📝 Generate Transcript
- 📋 Create Summary
- ✅ Extract Action Items
- 🔑 Extract Key Decisions
- ❓ Identify Questions
- 💬 Let you chat with the meeting
"""
)

st.divider()

# -------------------------------------------------
# INPUT SECTION
# -------------------------------------------------

col1, col2 = st.columns([1, 1])

with col1:

    input_type = st.radio(
        "Choose Source",
        [
            "Upload File",
            "YouTube URL"
        ]
    )

with col2:

    language = st.selectbox(
        "Language",
        [
            "english",
            "hinglish"
        ]
    )

source = None

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------

if input_type == "Upload File":

    uploaded_file = st.file_uploader(
        "📂 Drag & Drop Video/Audio Here",
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv",
            "webm",
            "mp3",
            "wav",
            "m4a"
        ]
    )

    if uploaded_file:

        temp_dir = tempfile.gettempdir()

        temp_path = os.path.join(
            temp_dir,
            uploaded_file.name
        )

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        source = temp_path

        st.success("File Uploaded Successfully")

        if uploaded_file.type.startswith("video"):
            st.video(uploaded_file)

        elif uploaded_file.type.startswith("audio"):
            st.audio(uploaded_file)

# -------------------------------------------------
# YOUTUBE INPUT
# -------------------------------------------------

else:

    source = st.text_input(
        "Paste YouTube URL"
    )

# -------------------------------------------------
# PROCESS BUTTON
# -------------------------------------------------

if st.button("🚀 Process Meeting"):

    if not source:

        st.warning("Please upload a file or provide a YouTube URL.")

    else:

        with st.spinner("Processing video..."):

            try:

                result = run_pipeline(
                    source,
                    language
                )

                st.session_state.result = result

                st.success("Processing Complete!")

            except Exception as e:

                st.error(f"Error: {str(e)}")

# -------------------------------------------------
# RESULTS
# -------------------------------------------------

if st.session_state.result:

    result = st.session_state.result

    st.divider()

    st.subheader("📌 Generated Title")

    st.info(result["title"])

    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "📋 Summary",
            "✅ Action Items",
            "🔑 Decisions",
            "❓ Questions",
            "📄 Transcript"
        ]
    )

    # ---------------- SUMMARY ----------------

    with tab1:

        st.markdown(result["summary"])

    # ---------------- ACTION ITEMS ----------------

    with tab2:

        st.markdown(result["action_items"])

    # ---------------- DECISIONS ----------------

    with tab3:

        st.markdown(result["decisions"])

    # ---------------- QUESTIONS ----------------

    with tab4:

        st.markdown(result["questions"])

    # ---------------- TRANSCRIPT ----------------

    with tab5:

        st.text_area(
            "Transcript",
            result["transcript"],
            height=500
        )

    st.divider()

    st.subheader("💬 Chat With Your Meeting")

    rag_chain = result["rag_chain"]

    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input(
        "Ask anything about this meeting..."
    )

    if prompt:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        try:

            answer = ask_question(
                rag_chain,
                prompt
            )

        except Exception as e:

            answer = f"Error: {str(e)}"

        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )




        
