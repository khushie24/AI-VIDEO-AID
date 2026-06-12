import gradio as gr
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

rag_chain_state = {"chain": None}


def run_pipeline(file, youtube_url, language):
    if file is not None:
        source = file.name
    elif youtube_url.strip():
        source = youtube_url.strip()
    else:
        return "⚠️ Please upload a file or enter a YouTube URL.", "", "", "", "", "", ""

    try:
        chunks = process_input(source)
        transcript = transcribe_all(chunks, language=language)
        title = generate_title(transcript)
        summary = summarize(transcript)
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)
        rag_chain_state["chain"] = build_rag_chain(transcript)
        return title, summary, action_items, decisions, questions, transcript, "✅ Processing complete!"
    except Exception as e:
        return f"❌ Error: {str(e)}", "", "", "", "", "", ""


def chat(message, history):
    if rag_chain_state["chain"] is None:
        return history + [[message, "⚠️ Please process a meeting first."]]
    try:
        answer = ask_question(rag_chain_state["chain"], message)
    except Exception as e:
        answer = f"❌ Error: {str(e)}"
    history.append([message, answer])
    return history


with gr.Blocks(title="AI Meeting & Video Assistant") as demo:

    gr.HTML('<h1 style="text-align:center;color:#4F46E5;">🎥 AI Meeting & Video Assistant</h1>')
    gr.HTML('<p style="text-align:center;color:#6b7280;">Upload a video/audio file or paste a YouTube URL to transcribe, summarise, and chat with your meeting.</p>')

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input")
            file_input = gr.File(
                label="Upload Video / Audio File",
                file_types=[".mp4", ".mov", ".avi", ".mkv", ".webm", ".mp3", ".wav", ".m4a"]
            )
            youtube_input = gr.Textbox(
                label="Or paste YouTube URL",
                placeholder="https://www.youtube.com/watch?v=..."
            )
            language_input = gr.Dropdown(
                choices=["english", "hinglish"],
                value="english",
                label="Transcription Language"
            )
            process_btn = gr.Button("🚀 Process Meeting", variant="primary", size="lg")
            status_box = gr.Textbox(label="Status", interactive=False)

        with gr.Column(scale=2):
            gr.Markdown("### 📌 Results")
            title_output = gr.Textbox(label="Generated Title", interactive=False)

            with gr.Tabs():
                with gr.Tab("📋 Summary"):
                    summary_output = gr.Markdown()
                with gr.Tab("✅ Action Items"):
                    action_output = gr.Markdown()
                with gr.Tab("🔑 Key Decisions"):
                    decisions_output = gr.Markdown()
                with gr.Tab("❓ Questions"):
                    questions_output = gr.Markdown()
                with gr.Tab("📄 Transcript"):
                    transcript_output = gr.Textbox(label="Full Transcript", lines=15, interactive=False)

    gr.Markdown("---")
    gr.Markdown("### 💬 Chat With Your Meeting")
    chatbot = gr.Chatbot(height=400)

    with gr.Row():
        chat_input = gr.Textbox(placeholder="Ask anything about this meeting...", label="Your question", scale=5)
        send_btn = gr.Button("Send", variant="primary", scale=1)

    process_btn.click(
        fn=run_pipeline,
        inputs=[file_input, youtube_input, language_input],
        outputs=[title_output, summary_output, action_output, decisions_output, questions_output, transcript_output, status_box]
    )

    send_btn.click(fn=chat, inputs=[chat_input, chatbot], outputs=chatbot).then(fn=lambda: "", outputs=chat_input)
    chat_input.submit(fn=chat, inputs=[chat_input, chatbot], outputs=chatbot).then(fn=lambda: "", outputs=chat_input)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
