"""Retrieval module: load index and search for relevant chunks."""

from __future__ import annotations

import json
import os
from typing import Any, List

import bm25s
from tqdm import tqdm

from src.models import IndexedChunk, MinimalSource

import re

def preprocess_for_bm25(text: str) -> str:
    """Split camelCase and snake_case for better BM25 token overlap."""
    # camelCase → camel Case
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # snake_case → snake case
    text = text.replace('_', ' ')
    return text

def load_index(index_dir: str) -> tuple[bm25s.BM25, List[IndexedChunk]]:
    """Load BM25 index and chunks from disk."""
    chunks_path = os.path.join(index_dir, "chunks.json")

    retriever = bm25s.BM25.load(index_dir)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    chunks = [IndexedChunk(**c) for c in chunks_data]

    return retriever, chunks


def _retrieve_indices(query: str, retriever: bm25s.BM25, k: int) -> List[int]:
    """Run BM25 retrieval and return the top-k chunk indices for a query."""
    # query_tokens = bm25s.tokenize(query)
    query_tokens = bm25s.tokenize(preprocess_for_bm25(query), stopwords="en")

    # bm25s types are not precise; use Any to avoid Pyright conflicts
    # retrieve() returns (documents, scores), each shaped (n_queries, k).
    # We only ever pass a single query here, so take row 0 of the
    # documents array to get the k document indices for *this* query.
    results: Any = retriever.retrieve(query_tokens, k=k)
    documents: Any = results[0]
    raw_result: Any = documents[0]

    # Normalize to List[int] regardless of return type (ndarray or list)
    if hasattr(raw_result, "tolist"):
        raw_list: Any = raw_result.tolist()
        indices: List[int] = [int(x) for x in raw_list]
    else:
        indices = [int(x) for x in raw_result]

    return indices


def search(
    query: str,
    retriever: bm25s.BM25,
    chunks: List[IndexedChunk],
    k: int = 10,
) -> List[MinimalSource]:
    """Search for top-k relevant chunks, returning minimal source references."""
    indices = _retrieve_indices(query, retriever, k)

    sources: List[MinimalSource] = []
    for idx in indices:
        if 0 <= idx < len(chunks):
            chunk = chunks[idx]
            sources.append(MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index,
            ))

    return sources


def search_full(
    query: str,
    retriever: bm25s.BM25,
    chunks: List[IndexedChunk],
    k: int = 10,
) -> List[IndexedChunk]:
    """Search for top-k relevant chunks, returning full chunks (with text).

    Useful for answer generation, which needs the chunk text rather than
    just the file/character references.
    """
    indices = _retrieve_indices(query, retriever, k)
    return [chunks[idx] for idx in indices if 0 <= idx < len(chunks)]


def search_dataset(
    queries: list[str],
    retriever: bm25s.BM25,
    chunks: list[IndexedChunk],
    k: int = 10,
) -> list[list[MinimalSource]]:
    """Batch search for multiple queries."""
    all_sources: list[list[MinimalSource]] = []

    for query in tqdm(queries, desc="Searching"):
        sources = search(query, retriever, chunks, k)
        all_sources.append(sources)

    return all_sources
