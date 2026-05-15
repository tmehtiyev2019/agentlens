"""
RAG agent — knowledge retrieval and grounding.

Runs in parallel with technical, market, risk, and cost agents. Retrieves
relevant context from the ChromaDB vector store for the input idea, then
synthesizes assumption and citation lists from retrieved chunks.

State fields read:  idea
State fields written: rag_output, grounded, errors (on failure)
"""

import os

import structlog
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.agents._critique import revise_addendum
from src.knowledge_base.vector_store import RetrievedChunk, VectorStore

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class RAGContext(BaseModel):
    retrieved_chunks: list[RetrievedChunk] = Field(
        description="All chunks returned across all queries (deduped by chunk_id)."
    )
    domain_chunks: list[RetrievedChunk] = Field(
        description="Subset of retrieved_chunks where source_type='domain'."
    )
    portfolio_chunks: list[RetrievedChunk] = Field(
        description="Subset of retrieved_chunks where source_type='portfolio'."
    )
    query_used: str = Field(
        description="The focused retrieval query generated from the input idea."
    )
    grounded: bool = Field(
        description="True when at least one chunk above the similarity threshold was retrieved."
    )
    total_retrieved: int = Field(
        description="Total number of chunks returned (all source types combined)."
    )
    assumption_list: list[str] = Field(
        description=(
            "Explicit assumptions that must hold for the retrieved evidence to be "
            "applicable to this idea."
        )
    )
    evidence_citations: list[str] = Field(
        description="Unique source_file values from all retrieved chunks."
    )


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_QUERY_SYSTEM_PROMPT = """\
You are a search query specialist. Given an early-stage idea, produce a single, \
focused retrieval query (1 sentence, no preamble) that will surface the most \
relevant technical and market grounding evidence from a research knowledge base.

Rules:
- Output only the query string — no explanation, no quotes, no labels.
- Focus on the core technical mechanism or market category, not the product framing.
- Be specific enough that a semantic search returns domain-relevant results.
"""

_ASSUMPTION_SYSTEM_PROMPT = """\
You are a critical analyst reviewing retrieved knowledge base excerpts. \
Given an idea and a set of retrieved document excerpts, produce a concise list \
of the key assumptions that must hold for the retrieved evidence to be validly \
applicable to this idea (e.g., technology maturity assumptions, market size \
assumptions, regulatory assumptions). Each assumption should be a single, \
falsifiable statement.

Output only a JSON array of strings. No explanation outside the array.
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_retrieval_query(idea: str, model_name: str) -> str:
    """
    Use a lightweight LLM call to compress the idea into a focused retrieval query.
    Temperature=0 for determinism; no tool use needed.
    """
    llm = ChatOpenAI(model=model_name, temperature=0, max_tokens=128)
    response = llm.invoke(
        [
            SystemMessage(content=_QUERY_SYSTEM_PROMPT),
            HumanMessage(content=idea),
        ]
    )
    return str(response.content).strip()


def _generate_assumptions(
    idea: str,
    chunks: list[RetrievedChunk],
    model_name: str,
) -> list[str]:
    """
    Ask the LLM to derive explicit assumptions from the retrieved chunks.
    Returns an empty list if chunk context is empty or parsing fails.
    """
    if not chunks:
        return []

    excerpts = "\n\n---\n\n".join(
        f"[{c.source_file}]\n{c.text[:600]}" for c in chunks[:8]
    )
    prompt = (
        f"Idea: {idea}\n\n"
        f"Retrieved excerpts:\n{excerpts}\n\n"
        "List the key assumptions (JSON array of strings):"
    )

    llm = ChatOpenAI(model=model_name, temperature=0, max_tokens=512)
    response = llm.invoke(
        [
            SystemMessage(content=_ASSUMPTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )
    raw = str(response.content).strip()

    import json

    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        logger.warning("assumption parsing failed — returning empty list", raw_preview=raw[:120])

    return []


def _dedup_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Deduplicate by chunk_id, keeping the highest-scoring copy."""
    seen: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        key = chunk.chunk_id or chunk.text[:64]
        if key not in seen or chunk.similarity_score > seen[key].similarity_score:
            seen[key] = chunk
    return list(seen.values())


# ---------------------------------------------------------------------------
# Agent entry point
# ---------------------------------------------------------------------------

def run(state: dict) -> dict:
    """
    Retrieve grounding context from ChromaDB for the input idea.

    Returns a partial GraphState update with rag_output and grounded.
    On VectorStore failure, returns rag_output=None and appends to errors.
    """
    idea: str = state["idea"] + revise_addendum(state)
    model_name = os.getenv("AGENT_MODEL", "gpt-4o-mini")

    log = logger.bind(agent="rag_agent", idea_preview=idea[:80])
    log.info("starting rag retrieval")

    try:
        vs = VectorStore()

        # Step 1: Generate a focused retrieval query from the idea
        query = _generate_retrieval_query(idea, model_name)
        log.info("retrieval query generated", query=query)

        # Step 2: Broad query — all source types
        all_chunks = vs.query(query_text=query)

        # Step 3: Source-type specific queries for structured breakdown
        domain_chunks = vs.query(query_text=query, source_type="domain")
        portfolio_chunks = vs.query(query_text=query, source_type="portfolio")

        # Deduplicate the union (broad query may overlap with typed queries)
        retrieved_chunks = _dedup_chunks(all_chunks)

        grounded = len(retrieved_chunks) > 0

        if not grounded:
            log.warning(
                "no chunks above similarity threshold — rag_output will be ungrounded",
                threshold=float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.75")),
            )

        # Step 4: Synthesize assumptions from retrieved evidence
        assumption_list = _generate_assumptions(idea, retrieved_chunks, model_name)

        # Step 5: Collect unique source file citations
        evidence_citations = sorted(
            {c.source_file for c in retrieved_chunks if c.source_file}
        )

        rag_context = RAGContext(
            retrieved_chunks=retrieved_chunks,
            domain_chunks=domain_chunks,
            portfolio_chunks=portfolio_chunks,
            query_used=query,
            grounded=grounded,
            total_retrieved=len(retrieved_chunks),
            assumption_list=assumption_list,
            evidence_citations=evidence_citations,
        )

        log.info(
            "rag retrieval complete",
            grounded=grounded,
            total_retrieved=len(retrieved_chunks),
            domain_count=len(domain_chunks),
            portfolio_count=len(portfolio_chunks),
            citations=evidence_citations,
        )

        return {"rag_output": rag_context, "grounded": grounded}

    except Exception as exc:  # noqa: BLE001
        log.exception("rag_agent failed", error=str(exc))
        return {
            "rag_output": None,
            "grounded": False,
            "errors": [f"rag_agent: {exc}"],
        }
