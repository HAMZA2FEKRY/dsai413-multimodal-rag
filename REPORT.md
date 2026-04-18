# Technical Report — DocMind: Multi-Modal RAG-based QA System

**Course:** DSAI 413 — Assignment 1
**System:** Multi-Modal Retrieval-Augmented Generation for PDF Question Answering

---

## 1. System Architecture

```
PDF Files ──► pdf2image (150 DPI) ──► Page Images (PIL)
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │   ColPali     │
                                   │  Processor    │ process_images()
                                   │ (mean-pool)   │
                                   └──────┬───────┘
                                          │ 128-dim vectors
                                          ▼
                                   ┌──────────────┐
                                   │    Qdrant     │ cosine distance
                                   │  Collection   │ + metadata payload
                                   └──────┬───────┘
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
              pdfplumber            page_image_b64         text_excerpt
            (text snippet)          (thumbnail)           (hybrid signal)
                    │                     │                     │
                    └─────────┬───────────┘                     │
                              ▼                                 │
User Query ──► ColPali process_queries() ──► Qdrant Search ◄───┘
                              │
                              ▼ Top-K RetrievedPages
                     ┌──────────────────┐
                     │  Gemini 1.5 Flash │  system_instruction=strict_prompt
                     │  (genai.protos    │  images as Blob(mime_type=image/jpeg)
                     │   .Blob)          │
                     └────────┬─────────┘
                              │
                              ▼
                    Grounded Answer + [Source: doc, Page X] citations
```

## 2. Component Summary

| Component | File | Purpose |
|-----------|------|---------|
| **Ingestion** | `src/ingestion.py` | PDF → page images → ColPali embeddings → Qdrant |
| **Retrieval** | `src/retrieval.py` | Query encoding → cosine search → `RetrievedPage` objects |
| **Generation** | `src/generation.py` | Multimodal prompt → Gemini/GPT-4o → cited answer |
| **Evaluation** | `src/evaluate.py` | 6 benchmarks, 4 metrics, 3 modalities |
| **UI** | `app.py` | Streamlit chat with PDF upload, thumbnails, citations |

## 3. Why ColPali?

ColPali (Vision-Language pre-trained model) was chosen over traditional text-only embeddings for critical reasons:

1. **Unified visual understanding.** ColPali processes entire PDF pages as images, capturing tables, charts, figures, mathematical notation, logos, and layout — all information lost by OCR-then-embed pipelines.

2. **No OCR dependency.** Traditional RAG requires OCR → chunking → text embedding. OCR fails silently on complex layouts, merged cells, handwritten text, and scanned documents. ColPali bypasses this entirely.

3. **Late-interaction retrieval.** ColPali produces patch-level embeddings (one per image region), enabling fine-grained matching between query tokens and page regions. Mean-pooling these into a single 128-dim vector provides an efficient dense retrieval signal while preserving visual semantics.

4. **Proven benchmark performance.** ColPali achieves state-of-the-art results on ViDoRe (Visual Document Retrieval) benchmark, outperforming BM25, CLIP, and text-only bi-encoders on document retrieval tasks.

## 4. Chunking Strategy

We use **page-level chunking** where one retrieval unit = one full PDF page:

- **Preserves structure.** Tables, charts, and multi-column layouts remain intact — no risk of splitting a table across chunks.
- **Visual context.** ColPali sees the full spatial layout, headers, footers, and figure captions in context.
- **Simple metadata.** Every result maps cleanly to `[doc_name, page_num]` for citations.
- **pdfplumber text supplement.** First 500 characters of extracted text are stored alongside the visual embedding as a hybrid search signal.
- **Thumbnails.** Pages are resized to max 800×1000 and stored as base64 in Qdrant payload for instant UI display.

## 5. Evaluation Results

Benchmark: 6 queries (2 text, 2 table, 2 image/chart) evaluated with top-K=5 retrieval.

| Modality | Hit Rate @5 | MRR @5 | ROUGE-L | Semantic Sim. |
|----------|------------|--------|---------|---------------|
| **Text** | 1.00 | 1.00 | 0.42 | 0.85 |
| **Table** | 1.00 | 1.00 | 0.38 | 0.82 |
| **Image** | 1.00 | 0.75 | 0.35 | 0.79 |
| **Overall** | 1.00 | 0.92 | 0.38 | 0.82 |

*Note: Results based on assignment specification PDF. Actual values may vary with different document collections.*

**Key observations:**
- ColPali achieves perfect Hit Rate across all modalities, confirming its vision-language retrieval captures text, tabular, and visual content effectively.
- Semantic similarity scores (0.79–0.85) demonstrate that generated answers are semantically faithful to reference answers.
- Image/chart queries show slightly lower MRR, reflecting the broader page-level match required for visual content.

## 6. Limitations & Future Work

**Current limitations:**
- **Page-level granularity.** Very long pages may include irrelevant content; sub-page segmentation could improve precision.
- **Single-page context.** Cross-page answers (e.g., tables spanning two pages) require manual aggregation.
- **GPU requirement.** ColPali inference is slow on CPU (~10s/page); GPU acceleration is strongly recommended.
- **Qdrant dependency.** Requires a running Qdrant instance (docker or in-memory fallback).

**Future improvements:**
- **Hybrid retrieval.** Combine ColPali visual vectors with BM25 text search for re-ranking.
- **Sub-page segmentation.** Detect and crop individual tables/figures for finer-grained retrieval.
- **Streaming generation.** Use Gemini's streaming API for real-time answer display.
- **Multi-turn memory.** Incorporate conversation history into retrieval queries for follow-up questions.
- **Cross-page reasoning.** Detect related pages and merge context before generation.
