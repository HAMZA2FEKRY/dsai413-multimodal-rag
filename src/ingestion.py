"""
src/ingestion.py
================
Multi-Modal PDF Ingestion & CLIP Indexing Pipeline
---------------------------------------------------
Converts every PDF page to a PIL image, embeds it using OpenAI CLIP
(512-dim vectors), and stores them in a Qdrant collection with
rich metadata (doc_name, page_num, text_excerpt, page_image_b64 thumbnail).

CLIP (Contrastive Language–Image Pre-training) embeds both images and text
into the same vector space, enabling cross-modal retrieval.
Model size: ~340 MB (vs ColPali's ~6 GB), fits comfortably in 4 GB RAM.

Usage (CLI):
    python -m src.ingestion --pdf_dir data/pdfs --index_name rag_index
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
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
CLIP_MODEL    = "openai/clip-vit-base-patch32"   # ~340 MB, fits in 4 GB RAM
QDRANT_HOST   = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT   = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDING_DIM = 512                     # CLIP ViT-B/32 output dimension
COLLECTION_PREFIX = "clip_"
MAX_TEXT_CHARS = 500                     # pdfplumber text snippet length
THUMB_MAX_SIZE = (800, 1000)             # max thumbnail dimensions

# Auto-detect bundled poppler on Windows (for pdf2image)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_POPPLER_BIN  = _PROJECT_ROOT / "poppler" / "poppler-24.08.0" / "Library" / "bin"
POPPLER_PATH  = str(_POPPLER_BIN) if _POPPLER_BIN.exists() else None


# ──────────────────────────────────────────────────────────────────────────────
# Windows safetensors mmap workaround
# ──────────────────────────────────────────────────────────────────────────────

def _apply_windows_safetensors_patch() -> None:
    """
    On Windows with a small paging file, safetensors uses memory-mapped I/O
    which raises OSError 1455 (paging file too small).

    Strategy: monkey-patch `safetensors.safe_open` with a pure-Python
    seeking reader (_SeekingSafeOpen) that:
      1. Reads only the JSON header (~KB) into RAM.
      2. For each `get_tensor()` call, seeks to the exact byte offset of that
         tensor's data and reads just those bytes — no mmap, no full-file read.
         Peak extra RAM ≈ size of the largest single tensor (~50–200 MB).
    """
    if sys.platform != "win32":
        return

    try:
        import json
        import struct

        import numpy as np
        import safetensors
        import torch

        # Safetensors dtype string → (numpy dtype, torch dtype)
        _DTYPE_MAP: dict = {
            "F64":  (np.float64,  torch.float64),
            "F32":  (np.float32,  torch.float32),
            "F16":  (np.float16,  torch.float16),
            "BF16": (None,        torch.bfloat16),   # numpy has no bf16
            "I64":  (np.int64,    torch.int64),
            "I32":  (np.int32,    torch.int32),
            "I16":  (np.int16,    torch.int16),
            "I8":   (np.int8,     torch.int8),
            "U8":   (np.uint8,    torch.uint8),
            "BOOL": (np.bool_,    torch.bool),
        }

        class _SeekingSafeOpen:
            """
            Reads individual tensors via file seek — avoids both mmap and
            loading the entire shard into RAM.
            """

            def __init__(self, path: str, framework: str, device: str = "cpu"):
                self._fh = open(path, "rb")
                # Parse 8-byte little-endian header length prefix
                header_len = struct.unpack("<Q", self._fh.read(8))[0]
                raw_header = self._fh.read(header_len)
                self._header: dict = json.loads(raw_header.decode("utf-8"))
                self._data_start = 8 + header_len
                self._tensor_keys = [k for k in self._header if k != "__metadata__"]
                logger.debug(
                    f"[win-patch] Opened {path} via seeking reader "
                    f"({len(self._tensor_keys)} tensors, no mmap)"
                )

            def __enter__(self):
                return self

            def __exit__(self, *_):
                self._fh.close()

            def keys(self):
                return self._tensor_keys

            def metadata(self) -> dict:
                """Return safetensors metadata block (transformers checks format='pt')."""
                return self._header.get("__metadata__", {"format": "pt"})

            def get_tensor(self, key: str):
                meta = self._header[key]
                dtype_str: str = meta["dtype"]
                shape: list = meta["shape"]
                start, end = meta["data_offsets"]
                num_bytes = end - start

                self._fh.seek(self._data_start + start)
                raw = self._fh.read(num_bytes)

                if dtype_str == "BF16":
                    # numpy has no bfloat16 — load as uint16 and reinterpret
                    arr = np.frombuffer(raw, dtype=np.uint16).reshape(shape)
                    return torch.from_numpy(arr.copy()).view(torch.bfloat16)

                np_dtype, torch_dtype = _DTYPE_MAP[dtype_str]
                arr = np.frombuffer(raw, dtype=np_dtype).reshape(shape)
                return torch.from_numpy(arr.copy()).to(torch_dtype)

        safetensors.safe_open = _SeekingSafeOpen  # type: ignore[assignment]

        # Also patch safetensors.torch.load_file — transformers calls this
        # directly after the metadata check (modeling_utils.py line 511).
        import safetensors.torch as st

        def _seeking_load_file(filename: str, device: str = "cpu") -> dict:
            """Load a safetensors file one tensor at a time without mmap."""
            result: dict = {}
            with _SeekingSafeOpen(filename, framework="pt", device=device) as f:
                for key in f.keys():
                    result[key] = f.get_tensor(key)
            return result

        st.load_file = _seeking_load_file  # type: ignore[assignment]
        logger.info("Applied Windows safetensors seeking (non-mmap) patch.")
    except Exception as exc:
        logger.warning(f"Could not apply safetensors Windows patch: {exc}")


# Apply once at module import time on Windows
_apply_windows_safetensors_patch()


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


def _load_clip_model(model_name: str = CLIP_MODEL) -> tuple:
    """
    Load the CLIP model and processor from HuggingFace transformers.

    CLIP ViT-B/32 is only ~340 MB and fits comfortably in 4 GB RAM.
    It embeds both images and text into the same 512-dim vector space.

    Returns
    -------
    (model, processor) tuple ready for embedding.
    """
    from transformers import CLIPModel, CLIPProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Try local cache first (avoids httpx version conflicts with HF Hub)
    _local_cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
    if _local_cache.exists():
        snapshots = _local_cache / "snapshots"
        local_dirs = sorted(snapshots.iterdir()) if snapshots.exists() else []
        if local_dirs:
            model_path = str(local_dirs[-1])
            logger.info(f"Loading CLIP model from local cache: {model_path}")
        else:
            model_path = model_name
            logger.info(f"Loading CLIP model: {model_name} on {device}")
    else:
        model_path = model_name
        logger.info(f"Loading CLIP model: {model_name} on {device}")

    model = CLIPModel.from_pretrained(model_path).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_path)

    logger.success(f"CLIP model loaded successfully on {device} (~340 MB RAM).")
    return model, processor


# ──────────────────────────────────────────────────────────────────────────────
# Core Ingestion Class
# ──────────────────────────────────────────────────────────────────────────────

class CLIPIngester:
    """
    Orchestrates the full ingestion pipeline:
      1. PDF → page images  (pdf2image at configurable DPI)
      2. Page images → CLIP embeddings  (512-dim vectors)
      3. Embeddings + metadata → Qdrant collection
    """

    def __init__(
        self,
        index_name: str,
        qdrant_host: str = QDRANT_HOST,
        qdrant_port: int = QDRANT_PORT,
        clip_model: str = CLIP_MODEL,
        dpi: int = 150,
        store_page_images: bool = True,
    ) -> None:
        """
        Initialise the ingester.

        Parameters
        ----------
        index_name       : logical name for this index (used as Qdrant collection suffix)
        qdrant_host/port : Qdrant connection details
        clip_model       : HuggingFace model ID for CLIP
        dpi              : rendering resolution for pdf2image (150 balances quality/speed)
        store_page_images: whether to store base64 thumbnails in Qdrant payload
        """
        self.index_name        = index_name
        self.collection_name   = COLLECTION_PREFIX + index_name
        self.dpi               = dpi
        self.store_page_images = store_page_images

        # Load CLIP model + processor
        self._model, self._processor = _load_clip_model(clip_model)
        self._device = next(self._model.parameters()).device

        # Connect to Qdrant — fall back to in-memory if server is unreachable
        logger.info(f"Connecting to Qdrant @ {qdrant_host}:{qdrant_port}")
        try:
            self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=5)
            self.qdrant.get_collections()  # test liveness
            logger.info("Connected to remote Qdrant.")
        except Exception as exc:
            logger.warning(
                f"Remote Qdrant unavailable ({exc}). "
                "Falling back to in-memory mode (data will not persist across restarts)."
            )
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
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
            )
        else:
            logger.info(f"Collection '{self.collection_name}' already exists — skipping creation.")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_image(self, img: Image.Image) -> np.ndarray:
        """
        Embed a single PIL image using CLIP's vision encoder.

        Steps:
            1. processor(images=[img]) → pixel_values
            2. model.get_image_features(**inputs) → [1, 512]
            3. L2-normalise → [512] unit vector
        """
        inputs = self._processor(images=img, return_tensors="pt").to(self._device)
        with torch.no_grad():
            features = self._model.get_image_features(**inputs)
        # L2-normalise for cosine similarity
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features[0].cpu().float().numpy()

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a text query using CLIP's text encoder."""
        inputs = self._processor(text=query, return_tensors="pt", truncation=True).to(self._device)
        with torch.no_grad():
            features = self._model.get_text_features(**inputs)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features[0].cpu().float().numpy()

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_pdf(self, pdf_path: Path) -> int:
        """
        Process a single PDF end-to-end:
          - Convert each page to an image at self.dpi
          - Embed with CLIP (512-dim)
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