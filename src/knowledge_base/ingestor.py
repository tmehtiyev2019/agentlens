"""
Document ingestion pipeline for the AgentLens knowledge base.

Reads files from disk, splits them into chunks, and upserts into ChromaDB
via VectorStore. Supports two source categories:

  domain    — static research / reference documents (src/knowledge_base/documents/)
  portfolio — historical project records (src/knowledge_base/portfolio/)
              with optional .meta.json sidecars per file

Supported file formats: .pdf, .md, .txt, .json

Deduplication: chunk IDs are deterministic (file_name + chunk_index). Because
VectorStore.add_documents uses IDs derived from chunk_id, Chroma upserts
rather than inserts — re-running ingest is safe and idempotent. A pre-flight
existence check using VectorStore.count() with a metadata query is also
performed to log skip counts, even though the upsert is already safe.

CLI usage:
    python -m src.knowledge_base.ingestor --ingest-domain
    python -m src.knowledge_base.ingestor --ingest-portfolio
    python -m src.knowledge_base.ingestor --list-sources
    python -m src.knowledge_base.ingestor --clear-portfolio
"""

import argparse
import hashlib
import json
import os
import pathlib
from typing import Any

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.knowledge_base.vector_store import VectorStore

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Config (all overridable via env vars)
# ---------------------------------------------------------------------------

_CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
_CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

_DOCUMENTS_DIR = pathlib.Path(__file__).parent / "documents"
_PORTFOLIO_DIR = pathlib.Path(__file__).parent / "portfolio"


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text(path: pathlib.Path) -> str:
    """
    Extract raw text from a supported file format.

    .pdf  → pypdf PdfReader, concatenate all page texts.
    .json → json.dumps (flatten to a single string so chunks stay coherent).
    .md / .txt / other text → read as UTF-8.
    """
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError(
                "pypdf is required for PDF ingestion. Install with: pip install pypdf"
            ) from exc
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if suffix == ".json":
        with path.open(encoding="utf-8", errors="replace") as fh:
            data: Any = json.load(fh)
        return json.dumps(data, ensure_ascii=False, indent=2)

    # .md, .txt, and any other text format
    return path.read_text(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _split_text(text: str) -> list[str]:
    """
    Split raw text into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter. Parameters come from env / config so they
    can be tuned without touching code.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=_CHUNK_SIZE,
        chunk_overlap=_CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_text(text)


# ---------------------------------------------------------------------------
# Chunk ID generation
# ---------------------------------------------------------------------------

def _make_chunk_id(source_file: str, chunk_index: int) -> str:
    """
    Generate a deterministic chunk ID from filename + index.

    Using a short SHA-256 prefix guards against IDs that are too long for
    Chroma's ID constraints while remaining collision-resistant for any
    realistic document set.
    """
    raw = f"{source_file}::{chunk_index}"
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"{source_file}__{chunk_index}__{digest}"


# ---------------------------------------------------------------------------
# Portfolio sidecar loader
# ---------------------------------------------------------------------------

def _load_portfolio_meta(path: pathlib.Path) -> dict:
    """
    Load optional <filename>.meta.json sidecar for a portfolio file.

    Expected sidecar structure:
        {
            "project_name": "...",
            "project_domain": "...",
            "outcome": "succeeded|killed|pivoted|unknown",
            "date_completed": "YYYY-MM-DD"
        }

    If the sidecar is missing or malformed, returns safe defaults so ingestion
    continues without interruption.
    """
    stem = path.stem  # e.g. "project_alpha" from "project_alpha.pdf"
    sidecar_path = path.parent / f"{stem}.meta.json"

    if sidecar_path.exists():
        try:
            with sidecar_path.open(encoding="utf-8") as fh:
                meta = json.load(fh)
            # Validate required fields — fall back to defaults for any missing key.
            return {
                "project_name":   meta.get("project_name", path.name),
                "project_domain": meta.get("project_domain", "unknown"),
                "outcome":        meta.get("outcome", "unknown"),
                "date_completed": meta.get("date_completed", ""),
            }
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "portfolio_sidecar_unreadable",
                sidecar=str(sidecar_path),
                error=str(exc),
            )

    return {
        "project_name":   path.name,
        "project_domain": "unknown",
        "outcome":        "unknown",
        "date_completed": "",
    }


# ---------------------------------------------------------------------------
# Core ingestion logic
# ---------------------------------------------------------------------------

def _ingest_files(
    directory: pathlib.Path,
    source_type: str,
    vs: VectorStore,
) -> int:
    """
    Ingest all supported files in `directory` into the vector store.

    Returns the total number of chunks successfully added.
    """
    _SUPPORTED = {".pdf", ".md", ".txt", ".json"}

    if not directory.exists():
        logger.warning("ingest_directory_missing", path=str(directory))
        return 0

    files = sorted(f for f in directory.iterdir() if f.is_file() and f.suffix.lower() in _SUPPORTED)
    if not files:
        logger.info("no_supported_files_found", directory=str(directory))
        return 0

    total_chunks_added = 0

    for path in files:
        # Skip sidecar files — they are metadata, not content documents.
        if path.name.endswith(".meta.json"):
            continue

        log = logger.bind(source_file=path.name, source_type=source_type)
        log.info("ingesting_file")

        try:
            raw_text = _extract_text(path)
        except Exception as exc:
            log.error("text_extraction_failed", error=str(exc))
            continue

        if not raw_text.strip():
            log.warning("empty_file_skipped")
            continue

        chunks = _split_text(raw_text)
        if not chunks:
            log.warning("no_chunks_produced")
            continue

        # Build per-chunk metadata
        suffix = path.suffix.lower()
        ext_to_doc_type = {".pdf": "pdf", ".md": "markdown", ".json": "json"}
        doc_type = ext_to_doc_type.get(suffix, "text")

        base_meta: dict = {
            "source_type": source_type,
            "source_file": path.name,
            "doc_type": doc_type,
        }
        if source_type == "portfolio":
            base_meta.update(_load_portfolio_meta(path))

        texts: list[str] = []
        metadatas: list[dict] = []
        skipped = 0

        for idx, chunk_text in enumerate(chunks):
            chunk_id = _make_chunk_id(path.name, idx)

            # Pre-flight deduplication check: query Chroma for this exact chunk_id.
            # Because VectorStore.add_documents upserts on ID, this check is
            # informational only — it lets us log skip counts without failing.
            try:
                existing = vs._store._collection.get(ids=[chunk_id])
                if existing and existing.get("ids"):
                    skipped += 1
                    continue
            except Exception as exc:
                # If the existence check fails, proceed with upsert — safe to do.
                log.debug("dedup_check_failed", chunk_id=chunk_id, error=str(exc))

            chunk_meta = {**base_meta, "chunk_id": chunk_id, "chunk_index": idx}
            texts.append(chunk_text)
            metadatas.append(chunk_meta)

        if skipped:
            log.info("chunks_skipped_already_exist", count=skipped)

        if not texts:
            log.info("all_chunks_already_ingested", file=path.name)
            continue

        try:
            vs.add_documents(texts=texts, metadatas=metadatas)
            total_chunks_added += len(texts)
            log.info(
                "file_ingested",
                new_chunks=len(texts),
                skipped_chunks=skipped,
                total_chunks=len(chunks),
            )
        except Exception as exc:
            log.error("add_documents_failed", error=str(exc), chunk_count=len(texts))

    return total_chunks_added


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentLens knowledge base ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m src.knowledge_base.ingestor --ingest-domain\n"
            "  python -m src.knowledge_base.ingestor --ingest-portfolio\n"
            "  python -m src.knowledge_base.ingestor --list-sources\n"
            "  python -m src.knowledge_base.ingestor --clear-portfolio\n"
        ),
    )
    parser.add_argument(
        "--ingest-domain",
        action="store_true",
        help=f"Ingest documents from {_DOCUMENTS_DIR}",
    )
    parser.add_argument(
        "--ingest-portfolio",
        action="store_true",
        help=f"Ingest documents from {_PORTFOLIO_DIR}",
    )
    parser.add_argument(
        "--list-sources",
        action="store_true",
        help="Print chunk counts by source_type",
    )
    parser.add_argument(
        "--clear-portfolio",
        action="store_true",
        help="Delete all portfolio chunks from the vector store",
    )
    args = parser.parse_args()

    if not any([args.ingest_domain, args.ingest_portfolio, args.list_sources, args.clear_portfolio]):
        parser.print_help()
        return

    vs = VectorStore()

    if args.list_sources:
        total = vs.count()
        domain = vs.count("domain")
        portfolio = vs.count("portfolio")
        print(f"Total chunks:     {total}")
        print(f"  domain:         {domain}")
        print(f"  portfolio:      {portfolio}")

    if args.clear_portfolio:
        before = vs.count("portfolio")
        vs.clear(source_type="portfolio")
        after = vs.count("portfolio")
        print(f"Cleared portfolio: {before} chunks removed ({after} remaining)")

    if args.ingest_domain:
        added = _ingest_files(
            directory=_DOCUMENTS_DIR,
            source_type="domain",
            vs=vs,
        )
        print(f"Domain ingestion complete: {added} new chunk(s) added from {_DOCUMENTS_DIR}")

    if args.ingest_portfolio:
        added = _ingest_files(
            directory=_PORTFOLIO_DIR,
            source_type="portfolio",
            vs=vs,
        )
        print(f"Portfolio ingestion complete: {added} new chunk(s) added from {_PORTFOLIO_DIR}")


if __name__ == "__main__":
    main()
