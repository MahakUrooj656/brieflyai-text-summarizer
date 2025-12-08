import os
import tempfile

import gradio as gr
from summarizer import TextSummarizer, compute_length_reduction


def build_model_name(choice: str) -> str:
    """
    Map a simple choice ('bart', 'distilbart', 't5') to a Hugging Face model id.
    """
    choice = choice.lower()
    if choice == "bart":
        return "facebook/bart-large-cnn"
    elif choice == "distilbart":
        return "sshleifer/distilbart-cnn-12-6"
    elif choice == "t5":
        return "t5-small"
    else:
        # default fallback
        return "facebook/bart-large-cnn"


def _write_summary_to_temp_file(summary_text: str) -> str | None:
    """
    Write the summary text to a temporary .txt file and return its path.
    This path is what DownloadButton expects in this Gradio version.
    """
    if not summary_text or not summary_text.strip():
        return None

    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, "summary.txt")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    return file_path


def summarize_interface(text, file_path, model_choice, max_len, min_len, creative_mode):
    """
    Main function used by the Gradio UI.
    Decides input source (file > text), runs the summarizer,
    returns summary text, stats, and an update to show/hide the download button.
    """
    # 1) Decide source of text: file has priority over text box
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text_content = f.read()
        except Exception as e:
            # Error case → hide download button
            return f"Error reading file: {e}", "", gr.update(visible=False)
    else:
        text_content = text or ""

    if not text_content.strip():
        # No valid input → hide download button
        return "Please enter some text or upload a .txt file.", "", gr.update(visible=False)

    # 2) Map model choice
    model_name = build_model_name(model_choice)

    # 3) Build summarizer with chosen settings
    ts = TextSummarizer(
        model_name=model_name,
        max_length=int(max_len),
        min_length=int(min_len),
        do_sample=bool(creative_mode)
    )

    # 4) Generate summary
    summary = ts.summarize(text_content)

    # 5) Compute stats
    orig_len, sum_len, reduction = compute_length_reduction(text_content, summary)
    stats = (
        f"Original length: {orig_len} words\n"
        f"Summary length:  {sum_len} words\n"
        f"Reduction:       {reduction * 100:.1f}%"
    )

    # 6) Just show the download button; file will be created when user clicks it
    download_visibility = gr.update(visible=True)

    return summary, stats, download_visibility


def generate_download_file(summary_text: str):
    """
    Called when the user clicks the download button.
    Creates a temp file from the summary text and returns its path.
    """
    file_path = _write_summary_to_temp_file(summary_text)
    if file_path is None:
        # If there's no summary, keep button as-is (no change)
        return gr.update()
    # This sets the DownloadButton's internal 'value' to the file path,
    # which triggers the browser to start a download.
    return file_path


def set_status():
    return "⏳ Summarizing…"


def clear_status():
    return ""


with gr.Blocks(title="BrieflyAI – Text Summarizer") as demo:
    # --- HEADER ---
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 10px;">
            <h1 style="margin-bottom: 0;">BrieflyAI</h1>
            <p style="margin-top: 4px; color: gray;">
                A transformer-powered text summarizer using BART, DistilBART, and T5 to turn long text into concise briefs.
            </p>
        </div>
        """
    )

    # Status line (shows "Summarizing…" while running)
    status_text = gr.Markdown("")

    # --- MAIN LAYOUT ---
    with gr.Row():
        # LEFT: Input panel
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 📥 Input")

                input_text = gr.Textbox(
                    lines=12,
                    label="Text input",
                    placeholder="Paste a long article, email, or report here..."
                )

                input_file = gr.File(
                    label="Or upload a .txt file",
                    file_types=["text"],
                    type="filepath"
                )

                gr.Markdown(
                    "_Note: If both text and file are provided, the **file** will be used._"
                )

        # RIGHT: Settings panel
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⚙️ Settings")

                model_choice = gr.Dropdown(
                    choices=["bart", "distilbart", "t5"],
                    value="bart",
                    label="Model"
                )

                # Static "cheat sheet" so user can compare models before choosing
                gr.Markdown(
                    """
**Model details:**

- **bart** – Full-sized encoder–decoder summarization model (`facebook/bart-large-cnn`).  
  Highest summary quality; heavier and slightly slower.

- **distilbart** – Distilled version of BART (`sshleifer/distilbart-cnn-12-6`).  
  Faster and lighter, with a small trade-off in quality.

- **t5** – Lightweight text-to-text model (`t5-small`).  
  Smallest and fastest; good for quick, general summaries.
                    """
                )

                max_len = gr.Slider(
                    minimum=30,
                    maximum=300,
                    value=120,
                    step=10,
                    label="Max summary length (tokens)"
                )
                min_len = gr.Slider(
                    minimum=10,
                    maximum=150,
                    value=40,
                    step=5,
                    label="Min summary length (tokens)"
                )

                creative_mode = gr.Checkbox(
                    value=False,
                    label="Creative mode (sampling on)"
                )

                gr.Markdown(
                    """
**Summary settings:**

- **Max / Min length** – Control how long the summary can be (in tokens, roughly like words).  
- **Creative mode** – Enables sampling; summaries may be more varied but slightly less deterministic.
                    """
                )

                summarize_btn = gr.Button("Summarize", variant="primary")

    # --- OUTPUT SECTION ---
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Group():
                gr.Markdown("### 📝 Summary")
                summary_output = gr.Textbox(
                    lines=12,
                    label="Summary",
                    show_label=False
                )
                # Download button: starts hidden, becomes visible after first summary
                download_btn = gr.DownloadButton(
                    label="Download summary as .txt",
                    visible=False
                )

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📊 Stats")
                stats_output = gr.Textbox(
                    lines=6,
                    label="Stats",
                    show_label=False
                )

    # --- WIRING: status + summarize chain ---
    summarize_btn.click(
        fn=set_status,
        inputs=None,
        outputs=status_text
    ).then(
        fn=summarize_interface,
        inputs=[input_text, input_file, model_choice, max_len, min_len, creative_mode],
        outputs=[summary_output, stats_output, download_btn]
    ).then(
        fn=clear_status,
        inputs=None,
        outputs=status_text
    )

    # --- WIRING: download button click ---
    download_btn.click(
        fn=generate_download_file,
        inputs=summary_output,
        outputs=download_btn
    )

if __name__ == "__main__":
    demo.launch()
