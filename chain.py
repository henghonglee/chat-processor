#!/usr/bin/env python3
"""
RAG pipeline bootstrap utilities.

Converts processed chat chunks (in `chats-processed/<chat_name>/chunk_*.txt`)
into LangChain `Document` objects with rich metadata.

Features
- Load documents for a specific chat or all chats
- Preserve chunk boundaries from preprocessed data
- Export to JSONL for downstream indexing

Usage examples
  python3 chain.py facebook-soldegen --export docs.jsonl
  python3 chain.py --all --export all_docs.jsonl
  python3 chain.py whatsapp-jiawei
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# LangChain imports
from langchain_core.documents import Document

from chunk import parse_timestamp  # Reuses robust timestamp parsing (unix or legacy whatsapp)


# ==========================
# Public API (top of file)
# ==========================

def load_documents_for_chat(base_dir: str, chat_name: str) -> List["Document"]:
    """Load processed chunks for a specific chat as LangChain Documents.

    Each chunk file becomes one Document.

    Metadata per Document includes:
    - chat_name: e.g., "facebook-soldegen"
    - chunk_id: filename like "chunk_001.txt"
    - source_path: absolute file path
    - message_count: int (lines starting with [timestamp])
    - ts_start / ts_end: unix timestamps (ints) when available
    - context: content from context.txt file if available
    """
    chat_dir = Path(base_dir).resolve() / "chats-processed" / chat_name
    if not chat_dir.exists():
        raise FileNotFoundError(f"Processed chat directory not found: {chat_dir}")

    # Load context.txt if it exists
    context_content = _load_context_file(chat_dir)

    chunk_files = sorted(chat_dir.glob("chunk_*.txt"))
    documents: List[Document] = []

    for chunk_file in chunk_files:
        page_text = _read_file_text(chunk_file)
        ts_start, ts_end, message_count = _analyze_chunk(page_text)

        metadata = {
            "chat_name": chat_name,
            "chunk_id": chunk_file.name,
            "message_count": message_count,
        }
        if ts_start is not None:
            metadata["ts_start"] = ts_start
        if ts_end is not None:
            metadata["ts_end"] = ts_end
        if context_content is not None:
            metadata["context"] = context_content

        document = Document(page_content=page_text, metadata=metadata)
        documents.append(document)

    return documents


def load_all_documents(base_dir: str) -> List["Document"]:
    """Load documents for all processed chats under `chats-processed/`."""
    processed_dir = Path(base_dir).resolve() / "chats-processed"
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    documents: List[Document] = []
    for chat_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
        documents.extend(
            load_documents_for_chat(
                base_dir=base_dir,
                chat_name=chat_dir.name,
            )
        )
    return documents


def export_documents_to_jsonl(documents: List["Document"], output_path: str) -> None:
    """Export a list of LangChain Documents to JSONL (page_content + metadata)."""
    output_file = Path(output_path).resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for doc in documents:
            record = {
                "page_content": doc.page_content,
                "metadata": dict(doc.metadata or {}),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load processed chunks into LangChain Documents and export if needed.")
    parser.add_argument("chat_name", nargs="?", help="Specific processed chat (folder name under chats-processed/)")
    parser.add_argument("--base-dir", default=".", help="Project base directory (contains chats-processed/)")
    parser.add_argument("--all", action="store_true", help="Load all processed chats")
    parser.add_argument("--export", help="Export documents to JSONL at the given path")

    args = parser.parse_args()

    if not args.all and not args.chat_name:
        parser.error("Provide a chat_name or use --all")

    if args.all:
        documents = load_all_documents(base_dir=args.base_dir)
    else:
        documents = load_documents_for_chat(
            base_dir=args.base_dir,
            chat_name=args.chat_name,
        )

    print(f"Loaded {len(documents)} document(s)")

    if args.export:
        export_documents_to_jsonl(documents, args.export)
        print(f"Exported to {Path(args.export).resolve()}")


# ==========================
# Private helpers (bottom)
# ==========================

def _read_file_text(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_context_file(chat_dir: Path) -> Optional[str]:
    """Load context.txt file content if it exists in the chat directory."""
    context_file = chat_dir / "context.txt"
    if context_file.exists():
        return _read_file_text(context_file)
    return None


def _analyze_chunk(chunk_text: str) -> Tuple[Optional[int], Optional[int], int]:
    """Extract first/last timestamps and message count from a chunk's raw text.

    Returns (ts_start, ts_end, message_count)
    where ts_* are unix epoch seconds when available.
    """
    lines = [line for line in chunk_text.splitlines() if line.strip()]
    timestamps = []
    message_count = 0

    for line in lines:
        # Only count message lines that begin with a timestamp bracket
        if line.startswith("["):
            message_count += 1
            dt = parse_timestamp(line)
            if dt is not None:
                timestamps.append(dt)

    if not timestamps:
        return None, None, message_count

    ts_start_dt = min(timestamps)
    ts_end_dt = max(timestamps)

    ts_start = int(ts_start_dt.timestamp())
    ts_end = int(ts_end_dt.timestamp())

    return ts_start, ts_end, message_count


if __name__ == "__main__":
    main()
