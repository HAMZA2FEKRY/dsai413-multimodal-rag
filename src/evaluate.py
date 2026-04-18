"""
src/evaluate.py
===============
Evaluation Suite
----------------
Benchmarks the RAG pipeline across three modality types:
  - TEXT  : questions answerable from paragraphs
  - TABLE : questions about tabular data
  - IMAGE : questions about charts/figures

Metrics computed per query:
  - Hit Rate @K      : was the correct page in the top-K results?  (1 or 0)
  - MRR @K           : Mean Reciprocal Rank of the first correct page
  - ROUGE-L          : lexical faithfulness of generated answer vs. reference
  - Semantic Sim.    : cosine similarity between answer and reference embeddings

Usage:
    python -m src.evaluate --index_name rag_index --top_k 5 --backend gemini
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from loguru import logger
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

from src.generation import get_generator
from src.retrieval import Retriever


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark dataset — 6 queries (2 per modality)
# ──────────────────────────────────────────────────────────────────────────────
# Each entry: query, expected_doc (partial match OK), expected_pages (set),
# reference_answer, modality

BENCHMARK_QUERIES: list[dict[str, Any]] = [
    # ── TEXT ──────────────────────────────────────────────────────────────────
    {
        "id":              "text_01",
        "modality":        "text",
        "query":           "What is the main objective of the document?",
        "expected_pages":  [1],
        "expected_doc":    None,    # any doc
        "reference_answer": (
            "The document aims to build a multi-modal retrieval-augmented generation "
            "system for answering questions from complex PDFs containing text, tables, "
            "charts, and images."
        ),
    },
    {
        "id":              "text_02",
        "modality":        "text",
        "query":           "What evaluation criteria are used to grade the assignment?",
        "expected_pages":  [2],
        "expected_doc":    None,
        "reference_answer": (
            "Criteria include accuracy and faithfulness (25%), multi-modal coverage "
            "(20%), system design (20%), innovation (15%), code quality (10%), "
            "and presentation (10%)."
        ),
    },
    # ── TABLE ─────────────────────────────────────────────────────────────────
    {
        "id":              "table_01",
        "modality":        "table",
        "query":           "List all deliverables required for the assignment.",
        "expected_pages":  [2],
        "expected_doc":    None,
        "reference_answer": (
            "Deliverables: (1) Codebase on GitHub, (2) Demo Application, "
            "(3) Technical Report (max 2 pages), (4) Video Demonstration (2–5 min)."
        ),
    },
    {
        "id":              "table_02",
        "modality":        "table",
        "query":           "What weight is assigned to System Design & Architecture?",
        "expected_pages":  [2],
        "expected_doc":    None,
        "reference_answer": "System Design & Architecture carries a weight of 20%.",
    },
    # ── IMAGE / CHART ─────────────────────────────────────────────────────────
    {
        "id":              "image_01",
        "modality":        "image",
        "query":           "Describe any visual diagrams or figures present in the document.",
        "expected_pages":  [1, 2],
        "expected_doc":    None,
        "reference_answer": (
            "The document contains a feature table listing expected system capabilities "
            "and an evaluation criteria table. These are structured visual elements "
            "embedded in the PDF."
        ),
    },
    {
        "id":              "image_02",
        "modality":        "image",
        "query":           "What does the expected features table describe?",
        "expected_pages":  [1],
        "expected_doc":    None,
        "reference_answer": (
            "The expected features table describes: multi-modal ingestion for text, "
            "tables and images; a vector index for unified embeddings; smart chunking; "
            "a QA chatbot; source attribution; and an evaluation suite."
        ),
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def hit_rate(retrieved_pages: list[int], expected_pages: list[int]) -> int:
    """
    Hit Rate @K: returns 1 if *any* expected page appears in the retrieved list,
    otherwise 0.
    """
    return int(bool(set(retrieved_pages) & set(expected_pages)))


def reciprocal_rank(retrieved_pages: list[int], expected_pages: list[int]) -> float:
    """
    MRR @K: returns 1/rank of the first relevant page in the retrieved list.
    Returns 0.0 if no relevant page is found.
    """
    expected_set = set(expected_pages)
    for rank, page in enumerate(retrieved_pages, start=1):
        if page in expected_set:
            return 1.0 / rank
    return 0.0


def rouge_l(prediction: str, reference: str) -> float:
    """Compute ROUGE-L F1 between prediction and reference strings."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure


# Lazy-loaded sentence transformer for semantic similarity
_sem_model: SentenceTransformer | None = None


def semantic_similarity(prediction: str, reference: str) -> float:
    """
    Compute cosine similarity between sentence-transformer embeddings
    using all-MiniLM-L6-v2 model.
    """
    global _sem_model
    if _sem_model is None:
        logger.info("Loading sentence-transformers model: all-MiniLM-L6-v2")
        _sem_model = SentenceTransformer("all-MiniLM-L6-v2")
    embs = _sem_model.encode([prediction, reference], normalize_embeddings=True)
    return float(np.dot(embs[0], embs[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class EvalResult:
    """Stores evaluation metrics for a single benchmark query."""

    query_id:        str
    modality:        str
    query:           str
    hit:             int
    rr:              float
    rouge_l:         float
    sem_sim:         float
    generated:       str
    reference:       str
    retrieved_pages: list[int] = field(default_factory=list)


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs the full evaluation pipeline:
      1. Retrieve top-K pages for each benchmark query
      2. Generate answers using the configured LLM backend
      3. Compute Hit Rate, MRR, ROUGE-L, and Semantic Similarity
      4. Aggregate by modality and overall
    """

    def __init__(
        self,
        index_name: str,
        top_k: int = 5,
        llm_backend: str = "gemini",
    ) -> None:
        """
        Initialise evaluator with retriever and generator.

        Parameters
        ----------
        index_name  : Qdrant index name (must match ingestion)
        top_k       : number of pages to retrieve per query
        llm_backend : "gemini" or "openai"
        """
        self.retriever = Retriever(index_name=index_name)
        self.generator = get_generator(llm_backend)
        self.top_k     = top_k

    def run(self, queries: list[dict[str, Any]] | None = None) -> list[EvalResult]:
        """
        Execute evaluation on all benchmark queries.

        Parameters
        ----------
        queries : override with custom queries (defaults to BENCHMARK_QUERIES)

        Returns
        -------
        List of EvalResult dataclasses, one per query.
        """
        queries = queries or BENCHMARK_QUERIES
        results: list[EvalResult] = []

        for q in queries:
            logger.info(f"Evaluating [{q['id']}] {q['query'][:60]}...")

            # Retrieve
            try:
                pages = self.retriever.search(q["query"], top_k=self.top_k)
                retrieved_page_nums = [p.page_num for p in pages]
            except Exception as exc:
                logger.error(f"Retrieval failed for {q['id']}: {exc}")
                retrieved_page_nums = []
                pages = []

            # Generate
            try:
                answer = self.generator.generate(q["query"], pages)
            except Exception as exc:
                logger.error(f"Generation failed for {q['id']}: {exc}")
                answer = ""

            # Score
            result = EvalResult(
                query_id=q["id"],
                modality=q["modality"],
                query=q["query"],
                hit=hit_rate(retrieved_page_nums, q["expected_pages"]),
                rr=reciprocal_rank(retrieved_page_nums, q["expected_pages"]),
                rouge_l=rouge_l(answer, q["reference_answer"]),
                sem_sim=semantic_similarity(answer, q["reference_answer"]),
                generated=answer,
                reference=q["reference_answer"],
                retrieved_pages=retrieved_page_nums,
            )
            results.append(result)
            logger.info(
                f"  hit={result.hit} | rr={result.rr:.2f} | "
                f"rouge_l={result.rouge_l:.2f} | sem_sim={result.sem_sim:.2f}"
            )

        return results

    @staticmethod
    def summarise(results: list[EvalResult]) -> dict[str, Any]:
        """
        Aggregate metrics by modality and overall.

        Returns
        -------
        dict with keys: "overall", "text", "table", "image"
        Each containing: hit_rate, mrr, rouge_l, sem_sim, n_queries
        """
        by_modality: dict[str, list[EvalResult]] = defaultdict(list)
        for r in results:
            by_modality[r.modality].append(r)

        def _agg(rs: list[EvalResult]) -> dict[str, float]:
            return {
                "hit_rate":    round(np.mean([r.hit     for r in rs]).item(), 4),
                "mrr":         round(np.mean([r.rr      for r in rs]).item(), 4),
                "rouge_l":     round(np.mean([r.rouge_l for r in rs]).item(), 4),
                "sem_sim":     round(np.mean([r.sem_sim for r in rs]).item(), 4),
                "n_queries":   len(rs),
            }

        summary: dict[str, Any] = {
            "overall": _agg(results),
        }
        for mod, rs in sorted(by_modality.items()):
            summary[mod] = _agg(rs)

        return summary


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI entry-point: python -m src.evaluate --index_name rag_index --top_k 5"""
    parser = argparse.ArgumentParser(description="RAG Evaluation Suite")
    parser.add_argument("--index_name", default="rag_index",
                        help="Qdrant index name (must match ingestion)")
    parser.add_argument("--top_k",      type=int, default=5,
                        help="Number of pages to retrieve per query")
    parser.add_argument("--backend",    default="gemini",
                        choices=["gemini", "openai"],
                        help="LLM backend for answer generation")
    parser.add_argument("--out",        default="data/eval_results.json",
                        help="Output path for evaluation results JSON")
    args = parser.parse_args()

    evaluator = Evaluator(
        index_name=args.index_name,
        top_k=args.top_k,
        llm_backend=args.backend,
    )
    results = evaluator.run()
    summary = evaluator.summarise(results)

    output = {
        "summary": summary,
        "results": [asdict(r) for r in results],
    }

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)

    logger.success(f"Evaluation complete. Results saved to {args.out}")
    logger.info("\n=== SUMMARY ===")
    for k, v in summary.items():
        logger.info(f"  [{k}] {v}")


if __name__ == "__main__":
    main()