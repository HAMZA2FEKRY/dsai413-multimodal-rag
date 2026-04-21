"""
src/generation.py
=================
Generation & QA Pipeline
-------------------------
Takes a user query + retrieved pages (images + text snippets) and sends them
to a Vision-Language Model (Gemini 1.5 Flash by default, GPT-4o as fallback)
to produce a grounded, citation-backed answer.

The system prompt strictly instructs the model to:
  - Answer ONLY from the provided document context.
  - Include citations in the format: [Source: DocName, Page X]
  - Reply "I don't have enough information..." when the context is insufficient.
"""

from __future__ import annotations

import base64
import io
import os
from enum import Enum
from typing import Any

from loguru import logger
from PIL import Image

from src.retrieval import RetrievedPage


# ──────────────────────────────────────────────────────────────────────────────
# LLM backend selection
# ──────────────────────────────────────────────────────────────────────────────

class LLMBackend(str, Enum):
    """Supported LLM backends for answer generation."""
    GEMINI = "gemini"
    OPENAI = "openai"
    GROQ   = "groq"


SYSTEM_PROMPT = """You are a precise document Q&A assistant.

RULES (non-negotiable):
1. Answer ONLY using the document pages provided to you in this conversation.
2. If the answer cannot be found in the provided pages, respond with:
   "I don't have enough information in the provided document context to answer this question."
3. Every factual claim in your answer MUST be followed by a citation in the exact format:
   [Source: <document_name>, Page <page_number>]
4. Do NOT speculate, hallucinate, or use external knowledge.
5. When a table or chart is relevant, describe what you observe in it and cite the page.
6. Keep your answers concise and well-structured.
"""


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _pil_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert a PIL image to raw bytes (JPEG format)."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL image to a base-64 string."""
    return base64.b64encode(_pil_to_bytes(img, fmt)).decode("utf-8")


# ──────────────────────────────────────────────────────────────────────────────
# Generator base
# ──────────────────────────────────────────────────────────────────────────────

class BaseGenerator:
    """Abstract base class for answer generators."""

    def generate(
        self,
        query: str,
        retrieved_pages: list[RetrievedPage],
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate an answer from query + retrieved context pages."""
        raise NotImplementedError


# ──────────────────────────────────────────────────────────────────────────────
# Gemini generator (google.genai SDK)
# ──────────────────────────────────────────────────────────────────────────────

class GeminiGenerator(BaseGenerator):
    """
    Uses Google Gemini 2.0 Flash for multi-modal answer generation.

    Set GEMINI_API_KEY in your environment (or .env file).
    Images are sent as inline Part objects with mime_type image/jpeg.
    """

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        """Initialise the Gemini generator with API key from environment."""
        from google import genai
        from google.genai import types

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY environment variable is not set. "
                "Get one free at https://aistudio.google.com/app/apikey"
            )

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.model_name = model_name
        logger.info(f"Gemini generator ready: {model_name}")

    def generate(
        self,
        query: str,
        retrieved_pages: list[RetrievedPage],
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Build a multimodal prompt with page images and text excerpts,
        then call Gemini to generate a grounded answer.
        """
        # Build the multimodal user message parts
        parts: list[Any] = []

        # 1. Attach each retrieved page image + annotation
        for page in retrieved_pages:
            parts.append(
                self._types.Part.from_text(
                    f"\n--- Document Context: {page.citation} "
                    f"(relevance score: {page.score:.2f}) ---\n"
                )
            )

            # Send image as inline data
            if page.page_image is not None:
                img_bytes = _pil_to_bytes(page.page_image)
                parts.append(
                    self._types.Part.from_bytes(
                        data=img_bytes,
                        mime_type="image/jpeg",
                    )
                )

            if page.text_excerpt:
                parts.append(
                    self._types.Part.from_text(f"[Text excerpt]: {page.text_excerpt}\n")
                )

        # 2. Append the user question
        parts.append(self._types.Part.from_text(f"\nUser Question: {query}"))

        # 3. Call the model
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=[self._types.Content(role="user", parts=parts)],
                config=self._types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    temperature=0.2,
                    max_output_tokens=1024,
                ),
            )
            answer = response.text.strip()
        except Exception as exc:
            exc_str = str(exc)
            logger.error(f"Gemini API error: {exc_str}")
            if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str or "quota" in exc_str.lower():
                answer = (
                    "⚠️ **Gemini API rate limit reached.** Your free-tier daily quota is exhausted.\n\n"
                    "**Fix:** Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey), "
                    "create a **new API key**, and update your `.env` file."
                )
            else:
                answer = f"An error occurred while generating the answer: {exc_str}"

        return answer


# ──────────────────────────────────────────────────────────────────────────────
# GPT-4o generator (fallback)
# ──────────────────────────────────────────────────────────────────────────────

class OpenAIGenerator(BaseGenerator):
    """
    Uses OpenAI GPT-4o for multi-modal answer generation.

    Set OPENAI_API_KEY in your environment.
    Images are sent as base64 data URLs in the chat completions API.
    """

    def __init__(self, model_name: str = "gpt-4o") -> None:
        """Initialise the OpenAI generator with API key from environment."""
        from openai import OpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY environment variable is not set."
            )

        self.client     = OpenAI(api_key=api_key)
        self.model_name = model_name
        logger.info(f"OpenAI generator ready: {model_name}")

    def generate(
        self,
        query: str,
        retrieved_pages: list[RetrievedPage],
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Build a vision chat completion request with page images and text,
        then call GPT-4o to generate a grounded answer.
        """
        # Build messages
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Inject prior conversation turns (if any)
        for turn in (chat_history or []):
            messages.append(turn)

        # Build user content (text + images)
        user_content: list[dict[str, Any]] = []

        for page in retrieved_pages:
            user_content.append({
                "type": "text",
                "text": (
                    f"\n--- Document Context: {page.citation} "
                    f"(score: {page.score:.2f}) ---\n"
                    + (f"[Text excerpt]: {page.text_excerpt}" if page.text_excerpt else "")
                ),
            })
            if page.page_image is not None:
                img_b64 = _pil_to_b64(page.page_image)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                })

        user_content.append({"type": "text", "text": f"\nUser Question: {query}"})
        messages.append({"role": "user", "content": user_content})

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,     # type: ignore[arg-type]
                max_tokens=1024,
                temperature=0.2,
            )
            answer = resp.choices[0].message.content or ""
        except Exception as exc:
            logger.error(f"OpenAI API error: {exc}")
            answer = "An error occurred while generating the answer. Please try again."

        return answer.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Groq generator (free, fast, Llama 3.3 70B)
# ──────────────────────────────────────────────────────────────────────────────

class GroqGenerator(BaseGenerator):
    """
    Uses Groq Cloud API with Llama 3.3 70B for answer generation.

    Groq is free with generous limits (30 req/min, 6000 tokens/min).
    Since Groq is text-only, we use the text excerpts from retrieved pages.
    Set GROQ_API_KEY in your environment (or .env file).
    Get a free key at: https://console.groq.com/keys
    """

    def __init__(self, model_name: str = "llama-3.3-70b-versatile") -> None:
        """Initialise the Groq generator with API key from environment."""
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY environment variable is not set. "
                "Get a free key at https://console.groq.com/keys"
            )

        self.client = Groq(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Groq generator ready: {model_name}")

    def generate(
        self,
        query: str,
        retrieved_pages: list[RetrievedPage],
        chat_history: list[dict[str, str]] | None = None,
    ) -> str:
        """
        Build a text prompt from retrieved page excerpts and call Groq.
        Groq uses an OpenAI-compatible API (chat completions).
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        # Inject prior conversation turns
        for turn in (chat_history or []):
            messages.append(turn)

        # Build context from retrieved pages (text-only for Groq)
        context_parts: list[str] = []
        for page in retrieved_pages:
            part = (
                f"\n--- Document Context: {page.citation} "
                f"(relevance score: {page.score:.2f}) ---\n"
            )
            if page.text_excerpt:
                part += f"{page.text_excerpt}\n"
            else:
                part += "[This page contains visual content such as charts, tables, or images]\n"
            context_parts.append(part)

        user_message = "".join(context_parts) + f"\nUser Question: {query}"
        messages.append({"role": "user", "content": user_message})

        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1024,
                temperature=0.2,
            )
            answer = resp.choices[0].message.content or ""
        except Exception as exc:
            logger.error(f"Groq API error: {exc}")
            answer = f"An error occurred while generating the answer: {exc}"

        return answer.strip()


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_generator(backend: str = "groq", **kwargs: Any) -> BaseGenerator:
    """
    Factory function to select the LLM backend.

    The backend can be set via:
      1. LLM_BACKEND environment variable (highest priority)
      2. ``backend`` argument

    Parameters
    ----------
    backend : "groq" (default) | "gemini" | "openai"

    Returns
    -------
    An initialised BaseGenerator subclass.
    """
    backend = (os.environ.get("LLM_BACKEND") or backend).lower()

    if backend == LLMBackend.GROQ:
        return GroqGenerator(**kwargs)
    elif backend == LLMBackend.GEMINI:
        return GeminiGenerator(**kwargs)
    elif backend == LLMBackend.OPENAI:
        return OpenAIGenerator(**kwargs)
    else:
        raise ValueError(f"Unknown LLM backend: '{backend}'. Choose 'groq', 'gemini', or 'openai'.")