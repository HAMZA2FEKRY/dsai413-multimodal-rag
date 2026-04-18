"""
demo_gradio.py
==============
Gradio-based demo UI for Multi-Modal RAG.
Designed for Google Colab or any environment with Gradio.

Usage:
    python demo_gradio.py
    # or in Colab: just run the cell
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr
from loguru import logger


def create_demo() -> gr.Blocks:
    """Build and return the Gradio Blocks interface."""

    # Lazy-loaded singletons
    _retriever = None
    _generator = None
    _ingester = None

    def get_retriever(index_name: str = "rag_index"):
        nonlocal _retriever
        if _retriever is None:
            from src.retrieval import Retriever
            _retriever = Retriever(index_name=index_name)
        return _retriever

    def get_gen(backend: str = "gemini"):
        nonlocal _generator
        if _generator is None:
            from src.generation import get_generator
            _generator = get_generator(backend=backend)
        return _generator

    def ingest_pdfs(files, index_name: str = "rag_index"):
        """Ingest uploaded PDF files."""
        nonlocal _ingester
        if not files:
            return "No files uploaded."

        from src.ingestion import ColPaliIngester
        if _ingester is None:
            _ingester = ColPaliIngester(index_name=index_name, store_page_images=True)

        results = []
        for f in files:
            try:
                n = _ingester.ingest_pdf(Path(f.name))
                results.append(f"✓ {Path(f.name).name}: {n} pages indexed")
            except Exception as exc:
                results.append(f"✗ {Path(f.name).name}: {exc}")

        return "\n".join(results)

    def answer_question(query: str, top_k: int, backend: str, history: list):
        """Retrieve + generate answer for a query."""
        if not query.strip():
            return history, history, []

        try:
            retriever = get_retriever()
            pages = retriever.search(query, top_k=top_k)
        except Exception as exc:
            logger.error(f"Retrieval error: {exc}")
            pages = []

        try:
            generator = get_gen(backend)
            answer = generator.generate(query, pages)
        except Exception as exc:
            logger.error(f"Generation error: {exc}")
            answer = f"Error: {exc}"

        # Build gallery images
        gallery_items = []
        for p in pages:
            if p.page_image is not None:
                gallery_items.append((p.page_image, f"{p.doc_name} p{p.page_num} (score: {p.score:.2f})"))

        history = history or []
        history.append((query, answer))
        return history, history, gallery_items

    # ── Build UI ──────────────────────────────────────────────────────────────
    with gr.Blocks(
        title="DocMind · Multi-Modal RAG",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css="""
        .gradio-container { max-width: 1200px !important; }
        h1 { background: linear-gradient(135deg, #60a5fa, #a78bfa);
             -webkit-background-clip: text; -webkit-text-fill-color: transparent;
             font-size: 2rem !important; }
        """
    ) as demo:
        gr.Markdown("# 🧠 DocMind · Multi-Modal RAG\nColPali retrieval + Gemini generation with citations")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(height=500, label="Chat")
                query_box = gr.Textbox(placeholder="Ask a question about your documents…",
                                       label="Your Question", lines=1)
                send_btn = gr.Button("🔍 Ask", variant="primary")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                backend_dd = gr.Dropdown(["gemini", "openai"], value="gemini", label="LLM Backend")
                top_k_sl = gr.Slider(1, 10, value=5, step=1, label="Top-K Pages")

                gr.Markdown("### 📄 Upload PDFs")
                file_upload = gr.File(file_count="multiple", file_types=[".pdf"], label="PDF Files")
                ingest_btn = gr.Button("⚡ Ingest", variant="secondary")
                ingest_output = gr.Textbox(label="Ingestion Status", interactive=False)

                gr.Markdown("### 📑 Retrieved Pages")
                gallery = gr.Gallery(label="Page Thumbnails", columns=2, height=300)

        state = gr.State([])

        send_btn.click(answer_question, [query_box, top_k_sl, backend_dd, state],
                       [chatbot, state, gallery])
        query_box.submit(answer_question, [query_box, top_k_sl, backend_dd, state],
                         [chatbot, state, gallery])
        ingest_btn.click(ingest_pdfs, [file_upload], [ingest_output])

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)
