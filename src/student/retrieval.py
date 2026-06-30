"""Retrieval module: load index and search for relevant chunks."""

from __future__ import annotations

import json
import os
import pickle
from typing import Any, List

import bm25s
from tqdm import tqdm

from student.models import IndexedChunk, MinimalSource


def load_index(index_dir: str) -> tuple[bm25s.BM25, List[IndexedChunk]]:
    """Load BM25 index and chunks from disk."""
    index_path = os.path.join(index_dir, "bm25_index.pkl")
    chunks_path = os.path.join(index_dir, "chunks.json")
    
    with open(index_path, "rb") as f:
        retriever = pickle.load(f)
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    chunks = [IndexedChunk(**c) for c in chunks_data]
    
    return retriever, chunks


def search(
    query: str,
    retriever: bm25s.BM25,
    chunks: List[IndexedChunk],
    k: int = 10,
) -> List[MinimalSource]:
    """Search for top-k relevant chunks."""
    query_tokens = bm25s.tokenize(query)
    
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


def search_dataset(
    queries: List[str],
    retriever: bm25s.BM25,
    chunks: List[IndexedChunk],
    k: int = 10,
) -> List[List[MinimalSource]]:
    """Batch search for multiple queries."""
    all_sources: List[List[MinimalSource]] = []
    
    for query in tqdm(queries, desc="Searching"):
        sources = search(query, retriever, chunks, k)
        all_sources.append(sources)
    
    return all_sources