"""
src/retrieval.py
================
Retrieval Pipeline
------------------
1. Encodes a user text query with ColPali's query processor.
2. Performs a nearest-neighbour (cosine) search in Qdrant.
3. Returns the top-K pages with full metadata (doc_name, page_num,
   text_excerpt, page thumbnail).

Usage:
    from src.retrieval import Retriever
    r = Retriever(index_name="rag_index")
    results = r.search("What does the chart on revenue growth show?", top_k=5)
"""

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger
from PIL import Image
from qdrant_client import QdrantClient


# ──────────────────────────────────────────────────────────────────────────────
# Defaults (mirror ingestion.py)
# ──────────────────────────────────────────────────────────────────────────────
COLPALI_MODEL     = "vidore/colpali-v1.2"
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_PREFIX = "colpali_"


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedPage:
    """A single retrieved document page with metadata and optional thumbnail."""

    doc_name:     str
    page_num:     int
    score:        float                         # cosine similarity [0, 1]
    text_excerpt: str = ""
    page_image:   Image.Image | None = None     # decoded thumbnail

    # ── Formatted citation string ──────────────────────────────────────────
    @property
    def citation(self) -> str:
        """Return a citation string in the format: [Source: DocName, Page X]"""
        return f"[Source: {self.doc_name}, Page {self.page_num}]"

    # ── Dict export (for JSON serialisation) ──────────────────────────────
    def to_dict(self, include_image: bool = False) -> dict[str, Any]:
        """Serialise this page to a dictionary (optionally including base64 image)."""
        d: dict[str, Any] = {
            "doc_name":     self.doc_name,
            "page_num":     self.page_num,
            "score":        round(float(self.score), 4),
            "text_excerpt": self.text_excerpt,
            "citation":     self.citation,
        }
        if include_image and self.page_image is not None:
            buf = io.BytesIO()
            self.page_image.save(buf, format="JPEG")
            d["page_image_b64"] = base64.b64encode(buf.getvalue()).decode()
        return d


# ──────────────────────────────────────────────────────────────────────────────
# Retriever
# ──────────────────────────────────────────────────────────────────────────────

class Retriever:
    """
    Thin wrapper around ColPali query encoding + Qdrant cosine search.

    Parameters
    ----------
    index_name     : matches the name used during ingestion
    colpali_model  : HuggingFace model ID for ColPali
    qdrant_host/port : location of your Qdrant instance
    """

    def __init__(
        self,
        index_name: str,
        colpali_model: str  = COLPALI_MODEL,
        qdrant_host: str    = QDRANT_HOST,
        qdrant_port: int    = QDRANT_PORT,
    ) -> None:
        """Initialise retriever: loads ColPali model and connects to Qdrant."""
        self.collection_name = COLLECTION_PREFIX + index_name

        # Load ColPali model + processor
        from colpali_engine.models import ColPali, ColPaliProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

        logger.info(f"Loading ColPali retrieval model: {colpali_model} on {device}")
        self._model = ColPali.from_pretrained(
            colpali_model,
            torch_dtype=dtype,
            device_map=device,
        ).eval()
        self._processor = ColPaliProcessor.from_pretrained(colpali_model)
        self._device = next(self._model.parameters()).device

        # Connect to Qdrant
        logger.info(f"Connecting to Qdrant @ {qdrant_host}:{qdrant_port}")
        try:
            self._qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        except Exception:
            logger.warning("Remote Qdrant unavailable, falling back to in-memory mode.")
            self._qdrant = QdrantClient(":memory:")

    # ── Query encoding ────────────────────────────────────────────────────────

    def _encode_query(self, query: str) -> np.ndarray:
        """
        Encode a text query with ColPali's query processor.

        Steps:
            1. processor.process_queries([query]) → model input batch
            2. model(**batch) → token embeddings [1, num_tokens, 128]
            3. mean-pool → [128] vector

        Returns
        -------
        np.ndarray of shape [128].
        """
        batch = self._processor.process_queries([query]).to(self._device)
        with torch.no_grad():
            embeddings = self._model(**batch)
        return embeddings[0].mean(dim=0).cpu().float().numpy()

    # ── Qdrant search ─────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> list[RetrievedPage]:
        """
        Embed ``query`` and return the top-K most similar document pages.

        Parameters
        ----------
        query           : natural-language user question
        top_k           : number of results to return
        score_threshold : minimum cosine similarity score to include

        Returns
        -------
        List[RetrievedPage] sorted by descending score.
        """
        logger.info(f"Query: '{query[:80]}...' | top_k={top_k}")

        try:
            vec = self._encode_query(query)
        except Exception as exc:
            logger.error(f"Query encoding failed: {exc}")
            return []

        try:
            hits = self._qdrant.search(
                collection_name=self.collection_name,
                query_vector=vec.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            logger.error(f"Qdrant search failed: {exc}")
            return []

        results: list[RetrievedPage] = []
        for hit in hits:
            payload = hit.payload or {}

            # Decode thumbnail if stored
            img: Image.Image | None = None
            if b64 := payload.get("page_image_b64"):
                try:
                    img_bytes = base64.b64decode(b64)
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as exc:             # noqa: BLE001
                    logger.warning(f"Image decode failed: {exc}")

            results.append(
                RetrievedPage(
                    doc_name=payload.get("doc_name", "unknown"),
                    page_num=int(payload.get("page_num", 0)),
                    score=float(hit.score),
                    text_excerpt=payload.get("text_excerpt", ""),
                    page_image=img,
                )
            )

        logger.success(f"Retrieved {len(results)} pages.")
        return results

    # ── Convenience: deduplicated search ─────────────────────────────────────

    def search_unique_docs(
        self, query: str, top_k: int = 5
    ) -> list[RetrievedPage]:
        """
        Like ``search``, but returns at most one page per document
        (the highest-scoring page for each doc).

        Useful for diverse multi-doc questions where you want coverage
        across different source PDFs.
        """
        raw = self.search(query, top_k=top_k * 3)   # over-fetch then dedupe
        seen: set[str] = set()
        unique: list[RetrievedPage] = []
        for r in raw:
            if r.doc_name not in seen:
                seen.add(r.doc_name)
                unique.append(r)
            if len(unique) >= top_k:
                break
        return unique