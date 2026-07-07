"""Indexing module: scan, chunk, and build BM25 index."""

from __future__ import annotations

import json
import bm25s
import os
from pathlib import Path

from tqdm import tqdm

from src.chunking import chunk_markdown, chunk_python
from src.models import IndexedChunk, MinimalSource


def scan_repository(repo_path: str) -> list[dict]:
    """Scan a repository for Python and Markdown files.
    
    Args:
        repo_path: Path to the repository root.
        
    Returns:
        List of file info dicts with 'path', 'content', 'type'.
    """
    repo = Path(repo_path)
    files = []
    
    # Find all .py and .md files recursively
    for pattern in ["**/*.py", "**/*.md"]:
        for file_path in repo.glob(pattern):
            # Skip hidden directories and __pycache__
            if any(part.startswith(".") or part == "__pycache__" 
                   for part in file_path.parts):
                continue
                
            try:
                content = file_path.read_text(encoding="utf-8")
                files.append({
                    "path": str(file_path),
                    "content": content,
                    "type": "python" if file_path.suffix == ".py" else "markdown",
                })
            except (UnicodeDecodeError, OSError):
                # Skip files we can't read
                continue
    
    return files


def chunk_files(files: list[dict], max_chunk_size: int) -> list[IndexedChunk]:
    """Chunk all files into searchable pieces.
    
    Args:
        files: Output from scan_repository().
        max_chunk_size: Maximum characters per chunk.
        
    Returns:
        List of validated Chunk objects.
    """
    all_chunks: list[IndexedChunk] = []
    
    for file_info in tqdm(files, desc="Chunking files"):
        path = file_info["path"]
        content = file_info["content"]
        file_type = file_info["type"]
        
        if file_type == "python":
            raw_chunks = chunk_python(content, max_chunk_size)
        else:
            raw_chunks = chunk_markdown(content, max_chunk_size)
        
        for raw in raw_chunks:
            chunk = IndexedChunk(
                file_path=path,
                text=raw["text"],
                first_character_index=raw["first_character_index"],
                last_character_index=raw["last_character_index"],
            )
            all_chunks.append(chunk)
    
    return all_chunks


def build_index(repo_path: str, output_dir: str, max_chunk_size: int = 2000) -> None:
    """Build and save BM25 index from repository.
    
    Args:
        repo_path: Path to repository root.
        output_dir: Where to save index and chunks.
        max_chunk_size: Maximum chunk size in characters.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scan
    print(f"Scanning {repo_path}...")
    files = scan_repository(repo_path)
    print(f"Found {len(files)} files")
    
    # 2. Chunk
    print("Chunking...")
    chunks = chunk_files(files, max_chunk_size)
    print(f"Created {len(chunks)} chunks")
    
    # 3. Build BM25 index
    print("Building BM25 index...")
    corpus = [chunk.text for chunk in chunks]
    # Note: deliberately NOT passing corpus= here. If a corpus is attached,
    # bm25s.retrieve() returns the matched documents/text instead of integer
    # doc indices, which breaks retrieval.search()'s index-based lookup.
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))
    
    # 4. Save everything
    chunks_path = os.path.join(output_dir, "chunks.json")
    
    retriever.save(output_dir)
    
    # Save chunks as list of dicts (Pydantic handles validation)
    chunks_data = [chunk.model_dump() for chunk in chunks]
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2)
    
    print(f"Index saved to {output_dir}")
    print(f"Chunks saved to {chunks_path}")