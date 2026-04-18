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
