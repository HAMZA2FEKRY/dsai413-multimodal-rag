"""
app.py
======
Multi-Modal RAG — Streamlit Chat Interface
------------------------------------------
Features:
  • PDF upload & on-the-fly CLIP ingestion
  • Chat interface with streaming-style display
  • Retrieved page thumbnails shown alongside each answer
  • Citation badges on every answer
  • Session-level conversation history
  • Dark theme with Sora font and gradient title

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="DocMind · Multi-Modal RAG",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Lazy imports (avoid slow load on startup) ─────────────────────────────────
@st.cache_resource(show_spinner="Loading CLIP retrieval model…")
def load_retriever(index_name: str):
    """Cache the Retriever instance to avoid reloading on every interaction."""
    from src.retrieval import Retriever
    return Retriever(index_name=index_name)


@st.cache_resource(show_spinner="Loading generation model…")
def load_generator(backend: str):
    """Cache the Generator instance to avoid re-initialising API clients."""
    from src.generation import get_generator
    return get_generator(backend=backend)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* ── Typography & palette ── */
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');
    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    /* ── Dark theme base ── */
    .stApp { background-color: #0a0e1a; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1117 0%, #111827 100%);
        border-right: 1px solid #1e293b;
    }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* ── Chat bubbles ── */
    .user-bubble {
        background: linear-gradient(135deg, #1e3a5f, #1a365d);
        border-radius: 16px 16px 4px 16px;
        padding: 12px 16px;
        margin: 8px 0;
        max-width: 75%;
        margin-left: auto;
        color: #e8f4fd;
        box-shadow: 0 2px 8px rgba(30, 58, 95, 0.3);
    }
    .assistant-bubble {
        background: linear-gradient(135deg, #1a1f2e, #1e2538);
        border: 1px solid #2d3748;
        border-radius: 4px 16px 16px 16px;
        padding: 14px 18px;
        margin: 8px 0;
        max-width: 88%;
        color: #d1d5db;
        font-size: 0.95rem;
        line-height: 1.7;
        box-shadow: 0 2px 8px rgba(26, 31, 46, 0.4);
    }

    /* ── Citation badge ── */
    .citation {
        display: inline-block;
        background: linear-gradient(135deg, #1a3a5c, #1e3f6b);
        border: 1px solid #2563eb;
        border-radius: 6px;
        padding: 2px 8px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: #60a5fa;
        margin: 2px 2px;
        transition: all 0.2s ease;
    }
    .citation:hover {
        background: #2563eb;
        color: #ffffff;
        transform: translateY(-1px);
    }

    /* ── Page thumbnail card ── */
    .thumb-card {
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 8px;
        text-align: center;
        background: linear-gradient(135deg, #111827, #0f172a);
        transition: transform 0.2s ease;
    }
    .thumb-card:hover {
        transform: scale(1.02);
        border-color: #4f46e5;
    }
    .thumb-caption {
        font-size: 0.7rem;
        color: #6b7280;
        margin-top: 4px;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* ── Score pill ── */
    .score-pill {
        background: linear-gradient(135deg, #14532d, #166534);
        border-radius: 12px;
        padding: 2px 8px;
        font-size: 0.7rem;
        color: #86efac;
        font-family: 'IBM Plex Mono', monospace;
    }

    /* ── Header ── */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-sub {
        color: #6b7280;
        font-size: 0.85rem;
        margin-top: 2px;
    }

    /* ── Ingest success animation ── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-8px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    .fade-in {
        animation: fadeIn 0.4s ease-out;
    }

    /* ── Scrollbar styling ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #374151; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #4b5563; }
</style>
""",
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Session state initialisation
# ──────────────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []          # [{role, content, pages}]
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs: list[str] = []


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — settings & upload
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocMind Settings")
    st.divider()

    index_name = st.text_input("Index name", value="rag_index",
                               help="Name of the Qdrant collection to query")
    llm_backend = st.selectbox("LLM Backend", ["gemini", "openai"], index=0,
                               help="Choose the LLM for answer generation")
    top_k = st.slider("Pages retrieved (top-K)", min_value=1, max_value=10, value=5,
                      help="Number of document pages to retrieve for each query")

    st.divider()
    st.markdown("### 📄 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drag & drop PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files and st.button("⚡ Ingest PDFs", use_container_width=True):
        from src.ingestion import CLIPIngester

        with st.spinner("Running CLIP indexing…"):
            try:
                ingester = CLIPIngester(
                    index_name=index_name,
                    store_page_images=True,
                )
                for uf in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(uf.read())
                        tmp_path = Path(tmp.name)
                    ingester.ingest_pdf(tmp_path)
                    st.session_state.ingested_docs.append(uf.name)
                    tmp_path.unlink(missing_ok=True)

                st.success(f"✓ Indexed {len(uploaded_files)} PDF(s)!")
            except (MemoryError, Exception) as exc:
                exc_str = str(exc)
                if "MemoryError" in type(exc).__name__ or "allocate" in exc_str.lower():
                    st.error(
                        "❌ **Out of RAM** — not enough free memory to load the model.\n\n"
                        "**Fix:** Close other applications and try again, "
                        "or run on Google Colab (see `colab_demo.ipynb`)."
                    )
                elif "1455" in exc_str or "paging file" in exc_str.lower():
                    st.error(
                        "❌ **Windows paging file too small.**\n\n"
                        "**Fix**: Control Panel → System → Advanced → Performance Settings → "
                        "Advanced → Change Virtual Memory → set to ≥ 16 GB."
                    )
                else:
                    st.error(f"Ingestion failed: {exc}")

    if st.session_state.ingested_docs:
        st.markdown("**Indexed documents:**")
        for doc in st.session_state.ingested_docs:
            st.markdown(f"- `{doc}`")

    st.divider()
    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown(
        "<small style='color:#4b5563'>DSAI 413 — Assignment 1<br>"
        "Multi-Modal RAG · ColPali + Gemini</small>",
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main area
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">DocMind · Multi-Modal RAG</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Ask any question about your uploaded PDFs — '
    "ColPali retrieves the most relevant pages, Gemini answers with citations.</div>",
    unsafe_allow_html=True,
)
st.divider()


def render_message(msg: dict) -> None:
    """Render a single chat message (user or assistant) with proper styling."""
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">💬 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        # Assistant answer
        st.markdown(
            f'<div class="assistant-bubble">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
        # Show retrieved page thumbnails in expander
        if pages := msg.get("pages"):
            with st.expander(f"📑 Retrieved pages ({len(pages)})", expanded=False):
                cols = st.columns(min(len(pages), 4))
                for idx, page in enumerate(pages):
                    with cols[idx % 4]:
                        if page.page_image is not None:
                            st.image(
                                page.page_image,
                                use_container_width=True,
                                caption=(
                                    f"{page.doc_name} · p{page.page_num} "
                                    f"· score {page.score:.2f}"
                                ),
                            )
                        else:
                            st.markdown(
                                f'<div class="thumb-card">'
                                f'<span style="font-size:2rem">📄</span>'
                                f'<div class="thumb-caption">'
                                f'{page.doc_name}<br>Page {page.page_num}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )


# Render conversation history
for msg in st.session_state.messages:
    render_message(msg)


# ── Input bar ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about your documents…")

if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    render_message(st.session_state.messages[-1])

    # Retrieve + generate
    with st.spinner("🔍 Searching document pages…"):
        try:
            retriever = load_retriever(index_name)
            pages = retriever.search(user_input, top_k=top_k)
        except Exception as exc:
            st.error(f"Retrieval error: {exc}")
            pages = []

    with st.spinner("✨ Generating answer…"):
        try:
            generator = load_generator(llm_backend)
            answer = generator.generate(user_input, pages)
        except Exception as exc:
            st.error(f"Generation error: {exc}")
            answer = "An error occurred while generating the answer."

    # Append assistant message
    asst_msg = {"role": "assistant", "content": answer, "pages": pages}
    st.session_state.messages.append(asst_msg)
    render_message(asst_msg)
    st.rerun()