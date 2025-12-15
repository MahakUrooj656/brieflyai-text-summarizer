# AI-Driven Text Summarizer
A small but powerful NLP project built in Python using Hugging Face Transformers.  
It summarizes long text (articles, reports, emails, notes) using transformer models like BART, DistilBART, and T5, and reports how much the text was compressed.

---

## Features

- **Transformer-based summarization** using Hugging Face `transformers`
- Support for multiple models:
  - `facebook/bart-large-cnn` (BART)
  - `sshleifer/distilbart-cnn-12-6` (DistilBART)
  - `t5-small` (T5)
- **CLI tool**:
  - Summarize raw text passed from the command line
  - Summarize `.txt` files
  - Configurable summary length (`--max`, `--min`)
  - Model selection via `--model`
  - Optional `--out` flag to write the summary to a file
- **Web UI (Gradio)**:
  - Paste text or upload a `.txt` file
  - Choose model from a dropdown
  - Control summary length with sliders
  - Toggle “creative mode” (sampling on/off)
  - Shows word-count statistics and % length reduction

---

## Tech Stack

- **Language:** Python
- **NLP Library:** Hugging Face `transformers`
- **Models:** BART, DistilBART, T5
- **Deep Learning Backend:** PyTorch (via `transformers[torch]`)
- **Web UI:** Gradio

---

## Evaluation

BrieflyAI separates inference from evaluation, following standard NLP practice.

- **Offline evaluation** is performed using ROUGE metrics on a small curated sample set with reference summaries.
- **Online inference** (CLI and Gradio app) operates on arbitrary user-provided text, where reference summaries are unavailable. For these cases, BrieflyAI reports compression statistics and supports qualitative inspection.

---
## Limitations
- ROUGE is computed only when a reference summary exists (offline or user-supplied).
- Small evaluation sample is not a substitute for full benchmark datasets.
- Summaries can occasionally miss details or rephrase in ways that reduce factual precision (common for abstractive models)

---

## Setup & Installation

```bash
# Clone this repository
git clone <your-repo-url>
cd ai_text_summarizer

# Create and activate virtual environment (optional but recommended)
python -m venv venv
# Windows:
# venv\Scripts\activate
# macOS / Linux:
# source venv/bin/activate

# Install dependencies
pip install "transformers[torch]" datasets tqdm gradio
