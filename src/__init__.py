"""
src — Multi-Modal RAG Pipeline
================================
Auto-loads environment variables from .env on import.
"""
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (two levels up from this file)
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)


# ── Shared Qdrant client (singleton) ─────────────────────────────────────────
# Both ingestion and retrieval MUST use the same client instance,
# otherwise in-memory mode creates separate empty databases.

import os
from loguru import logger

_QDRANT_CLIENT = None

def get_qdrant_client():
    """Return a shared QdrantClient instance (created once, reused everywhere)."""
    global _QDRANT_CLIENT
    if _QDRANT_CLIENT is not None:
        return _QDRANT_CLIENT

    from qdrant_client import QdrantClient

    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))

    try:
        client = QdrantClient(host=host, port=port, timeout=5)
        client.get_collections()  # test liveness
        logger.info(f"Connected to remote Qdrant @ {host}:{port}")
    except Exception as exc:
        logger.warning(f"Remote Qdrant unavailable ({exc}), using shared in-memory mode.")
        client = QdrantClient(":memory:")

    _QDRANT_CLIENT = client
    return _QDRANT_CLIENT
