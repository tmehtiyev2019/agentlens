"""
RAGAS metrics wrapper for the AgentLens RAG component.

Evaluates the RAG agent's output on four standard RAGAS metrics:
  - faithfulness:        does the answer use only facts from retrieved chunks?
  - answer_relevancy:   is the answer relevant to the input idea/query?
  - context_precision:  were the retrieved chunks actually useful for the answer?
  - context_recall:     did retrieval surface all documents relevant to the query?

Exported function:
    compute_rag_metrics(state: dict) -> RAGMetrics

Failure contract: any exception during RAGAS evaluation is caught, logged,
and returns a zero-score RAGMetrics — callers must not rely on this function
raising. This keeps the benchmark runner stable even when the RAGAS API is
unavailable or chunks are empty.
"""

import os

import structlog
from pydantic import BaseModel

logger = structlog.get_logger()

_FAITHFULNESS_THRESHOLD = float(os.getenv("RAGAS_FAITHFULNESS_MIN", "0.75"))
_CONTEXT_PRECISION_THRESHOLD = float(os.getenv("RAGAS_CONTEXT_PRECISION_MIN", "0.70"))


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class RAGMetrics(BaseModel):
    faithfulness: float        # 0.0–1.0: answer uses only facts from retrieved chunks
    answer_relevancy: float    # 0.0–1.0: answer is relevant to the idea query
    context_precision: float   # 0.0–1.0: retrieved chunks were actually useful
    context_recall: float      # 0.0–1.0: retrieval found all relevant documents
    passed_threshold: bool     # True iff faithfulness >= 0.75 AND context_precision >= 0.70
    grounded: bool             # mirrors state["rag_output"].grounded


def _zero_metrics(grounded: bool = False) -> RAGMetrics:
    return RAGMetrics(
        faithfulness=0.0,
        answer_relevancy=0.0,
        context_precision=0.0,
        context_recall=0.0,
        passed_threshold=False,
        grounded=grounded,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rag_metrics(state: dict) -> RAGMetrics:
    """
    Compute RAGAS metrics from the pipeline state after a full execution.

    Expects state["rag_output"] to be a RAGContext with:
        - .grounded: bool
        - .retrieved_chunks: list[RetrievedChunk], each with .text and .source_file
        - .synthesis: str — the synthesized answer the RAG agent produced

    If rag_output is absent, not grounded, or has empty chunks, returns
    zero-score RAGMetrics without calling RAGAS (avoids a pointless API call).

    The `ground_truth` field passed to RAGAS is set to state["idea"] because
    AgentLens does not maintain a human-labelled golden corpus. This makes
    context_recall an approximation — it measures whether the retrieval query
    (the idea) is well-covered by the retrieved chunks, not exact string recall.
    For a production evaluation harness, swap ground_truth for a curated answer set.
    """
    log = logger.bind(
        component="rag_metrics",
        idea_preview=str(state.get("idea", ""))[:80],
    )

    rag_output = state.get("rag_output")

    if rag_output is None:
        log.warning("rag_output absent from state — returning zero metrics")
        return _zero_metrics(grounded=False)

    grounded: bool = getattr(rag_output, "grounded", False)

    if not grounded:
        log.info("rag_output.grounded is False — skipping RAGAS evaluation")
        return _zero_metrics(grounded=False)

    chunks = getattr(rag_output, "retrieved_chunks", [])
    if not chunks:
        log.warning("rag_output has no retrieved_chunks — returning zero metrics")
        return _zero_metrics(grounded=True)

    idea: str = state.get("idea", "")
    synthesis: str = getattr(rag_output, "synthesis", "") or ""
    if not synthesis:
        # Fall back to concatenated chunk text if the agent did not produce a synthesis
        synthesis = " ".join(c.text for c in chunks if hasattr(c, "text"))

    contexts: list[str] = [c.text for c in chunks if hasattr(c, "text") and c.text]
    if not contexts:
        log.warning("all retrieved chunks have empty text — returning zero metrics")
        return _zero_metrics(grounded=True)

    try:
        # Import lazily so that importing this module does not force RAGAS to
        # initialise its default LLM/embedding clients at import time.
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict(
            {
                "question": [idea],
                "answer": [synthesis],
                "contexts": [contexts],
                # Using the idea as ground_truth is an intentional approximation.
                # See docstring for caveats.
                "ground_truth": [idea],
            }
        )

        log.info(
            "running ragas evaluation",
            chunk_count=len(contexts),
            synthesis_length=len(synthesis),
        )

        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )

        # RAGAS returns a dict-like object; access by metric name.
        faith_score = float(result["faithfulness"])
        relevancy_score = float(result["answer_relevancy"])
        precision_score = float(result["context_precision"])
        recall_score = float(result["context_recall"])

        passed = (
            faith_score >= _FAITHFULNESS_THRESHOLD
            and precision_score >= _CONTEXT_PRECISION_THRESHOLD
        )

        metrics = RAGMetrics(
            faithfulness=round(faith_score, 4),
            answer_relevancy=round(relevancy_score, 4),
            context_precision=round(precision_score, 4),
            context_recall=round(recall_score, 4),
            passed_threshold=passed,
            grounded=grounded,
        )

        log.info(
            "ragas evaluation complete",
            faithfulness=metrics.faithfulness,
            answer_relevancy=metrics.answer_relevancy,
            context_precision=metrics.context_precision,
            context_recall=metrics.context_recall,
            passed_threshold=metrics.passed_threshold,
        )
        return metrics

    except Exception as exc:
        log.error(
            "ragas evaluation failed",
            error=str(exc),
            chunk_count=len(contexts),
        )
        return _zero_metrics(grounded=grounded)
