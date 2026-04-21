# 🧠 DocMind — Multi-Modal RAG-based QA System

> **DSAI 413 — Assignment 1**
> A production-grade Multi-Modal Retrieval-Augmented Generation system that answers questions from complex real-world PDFs containing text, tables, charts, figures, scanned images, and footnotes.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Modal Ingestion** | Converts PDFs to page images, embeds with CLIP, stores in Qdrant |
| **Smart Chunking** | Page-level units preserve table structure, chart layout, and figure context |
| **Vision-Language Retrieval** | CLIP processes pages as images — no OCR pipeline needed |
| **Grounded Generation** | Gemini 1.5 Flash / GPT-4o generates answers with mandatory citations |
| **Interactive Chat UI** | Streamlit app with PDF upload, dark theme, page thumbnails |
| **Evaluation Suite** | Hit Rate, MRR, ROUGE-L, Semantic Similarity across text/table/image modalities |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Interface                           │
│                    (Streamlit / Gradio)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │ Query
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Retrieval Pipeline                            │
│      CLIP get_text_features() → Qdrant cosine search         │
└────────────────────────┬────────────────────────────────────────┘
                         │ Top-K RetrievedPages
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Generation Pipeline                           │
│    Gemini 1.5 Flash (images as Blob) / GPT-4o (base64 URLs)    │
│    System prompt enforces citation-only answers                 │
└────────────────────────┬────────────────────────────────────────┘
                         │ Grounded Answer + Citations
                         ▼
                    User sees answer + page thumbnails
```

---

## 📁 Project Structure

```
multimodal_rag/
├── app.py                  # Streamlit chat interface
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── README.md               # This file
├── REPORT.md               # Technical report (max 2 pages)
├── data/
│   ├── pdfs/               # PDF files for ingestion
│   ├── manifest.json        # Ingestion manifest
│   └── eval_results.json    # Evaluation output
└── src/
    ├── __init__.py          # Auto-loads .env
    ├── ingestion.py         # PDF → CLIP embeddings → Qdrant
    ├── retrieval.py         # Query encoding → Qdrant search → RetrievedPages
    ├── generation.py        # Multimodal prompt → Gemini/GPT-4o → cited answer
    └── evaluate.py          # Benchmark suite (6 queries, 4 metrics, 3 modalities)
```

---

## 🚀 Quick Start

### 1. Prerequisites

- **Python 3.10+**
- **Poppler** (required by pdf2image):
  - Windows: `conda install -c conda-forge poppler` or [download binaries](https://github.com/oschwartz10612/poppler-windows)
  - Linux: `sudo apt-get install poppler-utils`
  - macOS: `brew install poppler`
- **Qdrant** (local instance):
  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env and set your API keys:
#   GEMINI_API_KEY=your-key-here
#   OPENAI_API_KEY=your-key-here  (optional)
```

### 4. Ingest PDFs

```bash
# From a local directory:
python -m src.ingestion --pdf_dir data/pdfs --index_name rag_index

# Or download the Kaggle dataset first:
python -m src.ingestion --kaggle --index_name rag_index
```

### 5. Run the App

```bash
streamlit run app.py
```

### 6. Run Evaluation

```bash
python -m src.evaluate --index_name rag_index --top_k 5 --backend gemini
```

---

## 🔧 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GEMINI_API_KEY` | ✅ | — | Google AI Studio API key |
| `OPENAI_API_KEY` | ❌ | — | OpenAI key (fallback) |
| `LLM_BACKEND` | ❌ | `gemini` | `"gemini"` or `"openai"` |
| `QDRANT_HOST` | ❌ | `localhost` | Qdrant hostname |
| `QDRANT_PORT` | ❌ | `6333` | Qdrant port |
| `KAGGLE_USERNAME` | ❌ | — | For dataset download |
| `KAGGLE_KEY` | ❌ | — | For dataset download |

---

## 📊 Evaluation Metrics

The evaluation suite benchmarks the RAG pipeline across **3 modalities** (text, table, image) with **6 benchmark queries** (2 per modality):

| Metric | Description |
|--------|-------------|
| **Hit Rate @K** | Was the correct page in the top-K results? (1 or 0) |
| **MRR @K** | 1/rank of the first correct page |
| **ROUGE-L** | Lexical faithfulness vs. reference answer |
| **Semantic Similarity** | Cosine similarity via sentence-transformers (all-MiniLM-L6-v2) |

---

## 🧩 Tech Stack

| Component | Technology |
|-----------|-----------|
| Retrieval Model | CLIP ViT-B/32 (openai/clip-vit-base-patch32) |
| Vector Database | Qdrant (cosine distance, 512-dim) |
| Embedding | CLIPModel (image + text encoders, 512-dim) |
| Generation LLM | Gemini 1.5 Flash (primary) / GPT-4o (fallback) |
| PDF Processing | pdf2image (150 DPI) + pdfplumber (text extraction) |
| UI Framework | Streamlit (local) |
| Evaluation | rouge-score + sentence-transformers |
| Logging | Loguru |

---

## 📝 Citation Format

All answers follow the strict citation format:
```
[Source: document_name, Page X]
```

The system prompt prohibits hallucination and external knowledge — every claim must be backed by a retrieved document page.

---

## 📜 License

This project was built for DSAI 413 — Assignment 1.
