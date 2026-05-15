"""
ChromaDB abstraction layer for AgentLens.

All ingestion and retrieval flows through this module. Components (rag_agent,
ingestor scripts) import VectorStore and RetrievedChunk from here — never
interact with Chroma directly.

Design notes:
- Cosine similarity is enforced at collection-creation time via metadata.
- Similarity filtering (threshold) is applied post-query in Python because
  Chroma's `where_document` API does not support distance predicates.
- The `where` clause passed to Chroma uses the metadata field `source_type`
  exactly as stored; callers supply "domain" | "portfolio" | None.
"""

import os
import uuid
from typing import Any

import structlog
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from pydantic import BaseModel

logger = structlog.get_logger()

_COLLECTION_NAME = "agentlens_knowledge"
_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
_DEFAULT_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.75"))

_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class RetrievedChunk(BaseModel):
    text: str
    source_file: str
    source_type: str
    chunk_id: str
    similarity_score: float
    metadata: dict


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Thin wrapper around a persistent ChromaDB collection.

    All chunking happens upstream (in ingestor scripts). This class only
    embeds, stores, and retrieves pre-split text chunks.
    """

    def __init__(self) -> None:
        self._embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL)

        # Chroma collection-level cosine similarity is set via collection_metadata.
        # langchain_community.vectorstores.Chroma forwards collection_metadata to
        # the underlying chromadb client when the collection is first created.
        self._store = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=self._embeddings,
            persist_directory=_PERSIST_DIR,
            collection_metadata={"hnsw:space": "cosine"},
        )

        logger.info(
            "vector_store initialized",
            collection=_COLLECTION_NAME,
            persist_dir=_PERSIST_DIR,
            top_k=_DEFAULT_TOP_K,
            threshold=_SIMILARITY_THRESHOLD,
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_documents(self, texts: list[str], metadatas: list[dict]) -> None:
        """
        Embed and store pre-split text chunks.

        Each metadata dict must carry at minimum: source_type, source_file,
        chunk_id, doc_type. Portfolio documents additionally carry project_name,
        project_domain, outcome, date_completed.

        IDs are derived from chunk_id so re-ingesting the same file is
        idempotent — Chroma upserts on duplicate ID.
        """
        if len(texts) != len(metadatas):
            raise ValueError(
                f"texts and metadatas must have equal length "
                f"(got {len(texts)} vs {len(metadatas)})"
            )

        ids = [m.get("chunk_id", str(uuid.uuid4())) for m in metadatas]

        self._store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info(
            "documents added",
            count=len(texts),
            source_files=list({m.get("source_file", "unknown") for m in metadatas}),
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: str,
        top_k: int | None = None,
        source_type: str | None = None,
        outcome_filter: str | None = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most similar chunks above the similarity threshold.

        Parameters
        ----------
        query_text:
            Natural-language query string.
        top_k:
            Number of candidates to fetch from Chroma before threshold filtering.
            Defaults to RAG_TOP_K env var.
        source_type:
            When set to "domain" or "portfolio", adds a Chroma metadata filter
            so only documents of that type are searched.
        outcome_filter:
            When set (e.g., "succeeded"), further restricts portfolio results by
            the `outcome` metadata field. Requires source_type="portfolio".
        """
        k = top_k if top_k is not None else _DEFAULT_TOP_K

        where: dict[str, Any] | None = self._build_where(source_type, outcome_filter)

        try:
            results = self._store.similarity_search_with_relevance_scores(
                query=query_text,
                k=k,
                filter=where,
            )
        except Exception as exc:
            logger.error("chroma query failed", error=str(exc))
            raise

        chunks: list[RetrievedChunk] = []
        for doc, score in results:
            if score < _SIMILARITY_THRESHOLD:
                continue
            meta = doc.metadata or {}
            chunks.append(
                RetrievedChunk(
                    text=doc.page_content,
                    source_file=meta.get("source_file", ""),
                    source_type=meta.get("source_type", ""),
                    chunk_id=meta.get("chunk_id", ""),
                    similarity_score=round(score, 4),
                    metadata=meta,
                )
            )

        logger.info(
            "query complete",
            query_preview=query_text[:80],
            candidates=len(results),
            above_threshold=len(chunks),
            source_type=source_type,
            outcome_filter=outcome_filter,
        )
        return chunks

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count(self, source_type: str | None = None) -> int:
        """Return the number of stored chunks, optionally filtered by source_type."""
        collection = self._store._collection  # underlying chromadb Collection
        if source_type is None:
            return collection.count()
        results = collection.get(where={"source_type": source_type})
        return len(results["ids"])

    def clear(self, source_type: str | None = None) -> None:
        """
        Delete all chunks or, when source_type is given, only chunks of that type.

        Deleting by source_type uses Chroma's `where` filter on the collection.
        Deleting all recreates the collection to avoid stale HNSW index state.
        """
        collection = self._store._collection
        if source_type is None:
            all_ids = collection.get()["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
            logger.info("vector_store cleared", source_type="all", deleted=len(all_ids) if all_ids else 0)
        else:
            results = collection.get(where={"source_type": source_type})
            ids_to_delete = results["ids"]
            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
            logger.info(
                "vector_store cleared",
                source_type=source_type,
                deleted=len(ids_to_delete),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_where(
        source_type: str | None,
        outcome_filter: str | None,
    ) -> dict[str, Any] | None:
        conditions: list[dict] = []

        if source_type is not None:
            conditions.append({"source_type": {"$eq": source_type}})

        if outcome_filter is not None:
            conditions.append({"outcome": {"$eq": outcome_filter}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}


# ---------------------------------------------------------------------------
# CLI ingestor entry point
# ---------------------------------------------------------------------------
# Usage: python -m src.knowledge_base.vector_store --ingest
#
# Reads plain .txt files from src/knowledge_base/documents/ and upserts them
# into the collection as single-chunk documents (no splitting — documents
# directory contains pre-chunked seed files).

if __name__ == "__main__":
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="AgentLens knowledge base ingestor")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents/ into ChromaDB")
    parser.add_argument("--count", action="store_true", help="Print document counts")
    parser.add_argument("--clear", choices=["all", "domain", "portfolio"], help="Clear the store")
    args = parser.parse_args()

    vs = VectorStore()

    if args.count:
        print(f"Total:     {vs.count()}")
        print(f"Domain:    {vs.count('domain')}")
        print(f"Portfolio: {vs.count('portfolio')}")

    if args.clear:
        target = None if args.clear == "all" else args.clear
        vs.clear(target)
        print(f"Cleared: {args.clear}")

    if args.ingest:
        docs_dir = pathlib.Path(__file__).parent / "documents"
        if not docs_dir.exists():
            print(f"Documents directory not found: {docs_dir}")
        else:
            texts, metas = [], []
            for path in sorted(docs_dir.iterdir()):
                if not path.is_file():
                    continue
                suffix = path.suffix.lower()
                ext_map = {".pdf": "pdf", ".md": "markdown", ".json": "json"}
                doc_type = ext_map.get(suffix, "text")
                content = path.read_text(encoding="utf-8", errors="replace")
                chunk_id = f"{path.name}_0"
                texts.append(content)
                metas.append(
                    {
                        "source_type": "domain",
                        "source_file": path.name,
                        "chunk_id": chunk_id,
                        "doc_type": doc_type,
                    }
                )
            if texts:
                vs.add_documents(texts, metas)
                print(f"Ingested {len(texts)} document(s) from {docs_dir}")
            else:
                print("No files found in documents/")
