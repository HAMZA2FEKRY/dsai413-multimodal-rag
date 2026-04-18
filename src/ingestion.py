"""
src/ingestion.py
================
Multi-Modal PDF Ingestion & ColPali Indexing Pipeline
------------------------------------------------------
Converts every PDF page to a PIL image, embeds it using ColPali's processor
(mean-pooled 128-dim vectors), and stores them in a Qdrant collection with
rich metadata (doc_name, page_num, text_excerpt, page_image_b64 thumbnail).

Usage (CLI):
    python -m src.ingestion --pdf_dir data/pdfs --index_name rag_index
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pdfplumber
import torch
from loguru import logger
from pdf2image import convert_from_path
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Constants / defaults
# ──────────────────────────────────────────────────────────────────────────────
COLPALI_MODEL = "vidore/colpali-v1.2"
QDRANT_HOST   = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT   = int(os.getenv("QDRANT_PORT", "6333"))
COLPALI_DIM   = 128                     # ColPali mean-pooled vector dimension
COLLECTION_PREFIX = "colpali_"
MAX_TEXT_CHARS = 500                     # pdfplumber text snippet length
THUMB_MAX_SIZE = (800, 1000)             # max thumbnail dimensions

# Auto-detect bundled poppler on Windows (for pdf2image)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_POPPLER_BIN  = _PROJECT_ROOT / "poppler" / "poppler-24.08.0" / "Library" / "bin"
POPPLER_PATH  = str(_POPPLER_BIN) if _POPPLER_BIN.exists() else None


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL image to a base-64 string for metadata storage."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _extract_page_text(pdf_path: Path, page_num: int) -> str:
    """Return a short text excerpt (first 500 chars) from a page using pdfplumber."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # pdfplumber pages are 0-indexed
            page = pdf.pages[page_num - 1]
            text = page.extract_text() or ""
            return text[:MAX_TEXT_CHARS].strip()
    except Exception as exc:           # noqa: BLE001
        logger.warning(f"pdfplumber failed on {pdf_path} p{page_num}: {exc}")
        return ""


def _load_colpali_model(model_name: str = COLPALI_MODEL) -> tuple:
    """
    Load the ColPali model and processor from colpali_engine.

    Returns
    -------
    (model, processor) tuple ready for embedding.
    """
    from colpali_engine.models import ColPali, ColPaliProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float16 even on CPU to halve memory (~1.5GB instead of ~3GB)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    logger.info(f"Loading ColPali model: {model_name} on {device} ({dtype})")

    model = ColPali.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
        low_cpu_mem_usage=True,
    ).eval()

    processor = ColPaliProcessor.from_pretrained(model_name)

    logger.success(f"ColPali model loaded successfully on {device}.")
    return model, processor


# ──────────────────────────────────────────────────────────────────────────────
# Core Ingestion Class
# ──────────────────────────────────────────────────────────────────────────────

class ColPaliIngester:
    """
    Orchestrates the full ingestion pipeline:
      1. PDF → page images  (pdf2image at configurable DPI)
      2. Page images → ColPali embeddings  (colpali_processor.process_images)
      3. Embeddings + metadata → Qdrant collection
    """

    def __init__(
        self,
        index_name: str,
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT,
        colpali_model: str = COLPALI_MODEL,
        dpi: int = 150,
        store_page_images: bool = True,
    ) -> None:
        """
        Initialise the ingester.

        Parameters
        ----------
        index_name       : logical name for this index (used as Qdrant collection suffix)
        qdrant_host/port : Qdrant connection details
        colpali_model    : HuggingFace model ID for ColPali
        dpi              : rendering resolution for pdf2image (150 balances quality/speed)
        store_page_images: whether to store base64 thumbnails in Qdrant payload
        """
        self.index_name        = index_name
        self.collection_name   = COLLECTION_PREFIX + index_name
        self.dpi               = dpi
        self.store_page_images = store_page_images

        # Load ColPali model + processor
        self._model, self._processor = _load_colpali_model(colpali_model)
        self._device = next(self._model.parameters()).device

        # Connect to Qdrant
        logger.info(f"Connecting to Qdrant @ {qdrant_host}:{qdrant_port}")
        try:
            self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
        except Exception:
            logger.warning("Remote Qdrant unavailable, falling back to in-memory mode.")
            self.qdrant = QdrantClient(":memory:")

        self._ensure_collection()

    # ── Qdrant setup ─────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if it doesn't already exist."""
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in existing:
            logger.info(f"Creating Qdrant collection: {self.collection_name}")
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=COLPALI_DIM,
                    distance=Distance.COSINE,
                ),
            )
        else:
            logger.info(f"Collection '{self.collection_name}' already exists — skipping creation.")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_image(self, img: Image.Image) -> np.ndarray:
        """
        Embed a single PIL image using ColPali.

        Steps:
            1. processor.process_images([img]) → model input batch
            2. model(**batch) → patch embeddings [1, num_patches, 128]
            3. mean-pool → [128] vector
        """
        batch = self._processor.process_images([img]).to(self._device)
        with torch.no_grad():
            embeddings = self._model(**batch)
        # embeddings[0] shape: [num_patches, 128] → mean-pool → [128]
        vec = embeddings[0].mean(dim=0).cpu().float().numpy()
        return vec

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a text query using ColPali's query encoder."""
        batch = self._processor.process_queries([query]).to(self._device)
        with torch.no_grad():
            embeddings = self._model(**batch)
        vec = embeddings[0].mean(dim=0).cpu().float().numpy()
        return vec

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: Path) -> int:
        """
        Process a single PDF end-to-end:
          - Convert each page to an image at self.dpi
          - Embed with ColPali (mean-pooled 128-dim)
          - Upsert into Qdrant with metadata

        Parameters
        ----------
        pdf_path : path to the PDF file

        Returns
        -------
        Number of pages successfully ingested.
        """
        pdf_path = Path(pdf_path)
        doc_name = pdf_path.stem
        logger.info(f"Ingesting: {pdf_path.name}")

        try:
            pages: list[Image.Image] = convert_from_path(
                str(pdf_path), dpi=self.dpi, poppler_path=POPPLER_PATH
            )
        except Exception as exc:
            logger.error(f"pdf2image failed for {pdf_path.name}: {exc}")
            raise

        points: list[PointStruct] = []

        # Determine starting ID by checking existing points count
        try:
            existing_count = self.qdrant.count(self.collection_name).count
        except Exception:
            existing_count = 0

        for i, img in enumerate(tqdm(pages, desc=f"  Embedding {doc_name}", unit="page")):
            page_num = i + 1

            try:
                vec = self._embed_image(img)
            except Exception as exc:
                logger.error(f"Embedding failed for {doc_name} p{page_num}: {exc}")
                continue

            payload: dict[str, Any] = {
                "doc_name":     doc_name,
                "page_num":     page_num,
                "text_excerpt": _extract_page_text(pdf_path, page_num),
            }

            if self.store_page_images:
                # Store thumbnail for UI display (resize to save space)
                thumb = img.copy()
                thumb.thumbnail(THUMB_MAX_SIZE)
                payload["page_image_b64"] = _pil_to_b64(thumb)

            points.append(
                PointStruct(
                    id=existing_count + i,
                    vector=vec.tolist(),
                    payload=payload,
                )
            )

        if points:
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

        logger.success(f"  ✓ {len(points)} pages indexed for '{doc_name}'.")
        return len(points)

    def ingest_directory(self, pdf_dir: Path) -> dict[str, int]:
        """
        Ingest all PDFs inside a directory (recursive).

        Returns
        -------
        dict mapping filename → number of pages indexed (-1 if failed).
        """
        pdf_dir = Path(pdf_dir)
        pdfs    = sorted(pdf_dir.glob("**/*.pdf"))

        if not pdfs:
            logger.warning(f"No PDFs found in {pdf_dir}")
            return {}

        logger.info(f"Found {len(pdfs)} PDFs in {pdf_dir}")
        results: dict[str, int] = {}

        for pdf in pdfs:
            try:
                n = self.ingest_pdf(pdf)
                results[pdf.name] = n
            except Exception as exc:              # noqa: BLE001
                logger.error(f"Failed to ingest {pdf.name}: {exc}")
                results[pdf.name] = -1

        total = sum(v for v in results.values() if v > 0)
        logger.success(f"Ingestion complete. Total pages indexed: {total}")
        return results

    def save_manifest(
        self,
        results: dict[str, int],
        out_path: Path = Path("data/manifest.json"),
    ) -> None:
        """Persist ingestion summary for debugging / reproducibility."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(
                {
                    "index_name":  self.index_name,
                    "collection":  self.collection_name,
                    "colpali_dim": COLPALI_DIM,
                    "dpi":         self.dpi,
                    "documents":   results,
                },
                f, indent=2,
            )
        logger.info(f"Manifest saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Kaggle dataset helper
# ──────────────────────────────────────────────────────────────────────────────

def download_kaggle_dataset(dest: Path = Path("data/pdfs")) -> Path:
    """
    Download the Kaggle PDF dataset used for this assignment.
    Requires KAGGLE_USERNAME and KAGGLE_KEY env-vars (or ~/.kaggle/kaggle.json).

    Returns
    -------
    Path to the downloaded dataset directory.
    """
    try:
        import kagglehub
        path = kagglehub.dataset_download("manisha717/dataset-of-pdf-files")
        logger.success(f"Kaggle dataset downloaded to: {path}")
        return Path(path)
    except Exception as exc:
        logger.error(f"Kaggle download failed: {exc}")
        raise


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry-point: python -m src.ingestion --pdf_dir data/pdfs --index_name rag_index"""
    parser = argparse.ArgumentParser(description="ColPali PDF Ingestion Pipeline")
    parser.add_argument("--pdf_dir",    default="data/pdfs",   help="Directory of PDF files")
    parser.add_argument("--index_name", default="rag_index",   help="Index / collection name")
    parser.add_argument("--dpi",        type=int, default=150, help="PDF render DPI (higher = slower but better)")
    parser.add_argument("--kaggle",     action="store_true",   help="Download Kaggle dataset first")
    parser.add_argument("--no_images",  action="store_true",   help="Don't store page image thumbnails in Qdrant")
    args = parser.parse_args()

    if args.kaggle:
        pdf_dir = download_kaggle_dataset(Path(args.pdf_dir))
    else:
        pdf_dir = Path(args.pdf_dir)

    ingester = ColPaliIngester(
        index_name=args.index_name,
        dpi=args.dpi,
        store_page_images=not args.no_images,
    )
    results = ingester.ingest_directory(pdf_dir)
    ingester.save_manifest(results)


if __name__ == "__main__":
    main()