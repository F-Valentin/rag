"""Chunking strategies for Python and Markdown files."""

from __future__ import annotations

import ast
import re
from typing import List


def chunk_markdown(text: str, max_chunk_size: int) -> list[dict]:
    """Split markdown text into chunks by headers, then by size.
    
    Args:
        text: The markdown content.
        max_chunk_size: Maximum characters per chunk.
        
    Returns:
        List of chunk dicts with text, first_character_index, last_character_index.
    """
    if not text.strip():
        return []
    
    # Split on headers (#, ##, ###, etc.) but keep the delimiter
    pattern = re.compile(r'(?=^#{1,6} )', re.MULTILINE)
    sections = pattern.split(text)
    
    chunks = []
    current_pos = 0
    
    for section in sections:
        # Skip empty sections (can happen if text starts with a header)
        if not section.strip():
            current_pos += len(section)
            continue
        
        section_start = text.find(section, current_pos)
        if section_start == -1:
            section_start = current_pos
        
        if len(section) <= max_chunk_size:
            chunks.append({
                "text": section,
                "first_character_index": section_start,
                "last_character_index": section_start + len(section),
            })
        else:
            # Split oversized section into fixed-size chunks
            for i in range(0, len(section), max_chunk_size):
                chunk_text = section[i:i + max_chunk_size]
                chunk_start = section_start + i
                chunks.append({
                    "text": chunk_text,
                    "first_character_index": chunk_start,
                    "last_character_index": chunk_start + len(chunk_text),
                })
        
        current_pos = section_start + len(section)
    
    return chunks


def chunk_python(source: str, max_chunk_size: int) -> list[dict]:
    """Split Python code into chunks by top-level definitions.
    
    Args:
        source: The Python source code.
        max_chunk_size: Maximum characters per chunk.
        
    Returns:
        List of chunk dicts with text, first_character_index, last_character_index.
    """
    if not source.strip():
        return []
    
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # If the file has syntax errors, fall back to simple line-based chunking
        return _chunk_by_lines(source, max_chunk_size)
    
    chunks = []
    
    # Get top-level functions, classes, and async functions
    top_level = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    
    # If no top-level definitions, chunk by lines
    if not top_level:
        return _chunk_by_lines(source, max_chunk_size)
    
    for node in top_level:
        # Safely get line numbers
        start_line = getattr(node, 'lineno', 1)
        end_line = getattr(node, 'end_lineno', start_line)
        
        if end_line is None:
            end_line = start_line
        
        start_char = _line_to_char(source, start_line)
        end_char = _line_to_char(source, end_line + 1)
        
        # Ensure we don't go past the source length
        end_char = min(end_char, len(source))
        
        text = source[start_char:end_char]
        
        if len(text) <= max_chunk_size:
            chunks.append({
                "text": text,
                "first_character_index": start_char,
                "last_character_index": end_char,
            })
        else:
            # Split oversized definition into fixed-size chunks
            for i in range(0, len(text), max_chunk_size):
                chunk_text = text[i:i + max_chunk_size]
                chunk_start = start_char + i
                chunks.append({
                    "text": chunk_text,
                    "first_character_index": chunk_start,
                    "last_character_index": chunk_start + len(chunk_text),
                })
    
    return chunks


def _line_to_char(source: str, lineno: int) -> int:
    """Convert 1-based line number to 0-based character index.
    
    Args:
        source: The full source text.
        lineno: 1-based line number (line 1 = first line).
        
    Returns:
        Character index of the start of the given line.
        If lineno > number of lines, returns len(source).
    """
    lines = source.splitlines(keepends=True)
    
    if lineno <= 1:
        return 0
    if lineno > len(lines) + 1:
        return len(source)
    
    return sum(len(line) for line in lines[:lineno - 1])


def _chunk_by_lines(source: str, max_chunk_size: int) -> list[dict]:
    """Fallback chunking: split by lines, respecting line boundaries.
    
    Args:
        source: The source text.
        max_chunk_size: Maximum characters per chunk.
        
    Returns:
        List of chunk dicts with text, first_character_index, last_character_index.
    """
    lines = source.splitlines(keepends=True)
    chunks = []
    
    current_lines: List[str] = []
    current_length = 0
    current_start = 0
    
    for line in lines:
        line_len = len(line)
        
        # If adding this line exceeds max_chunk_size, flush current chunk
        if current_length + line_len > max_chunk_size and current_lines:
            chunk_text = "".join(current_lines)
            chunks.append({
                "text": chunk_text,
                "first_character_index": current_start,
                "last_character_index": current_start + current_length,
            })
            # Start new chunk
            current_lines = [line]
            current_length = line_len
            current_start += len(chunk_text)
        else:
            current_lines.append(line)
            current_length += line_len
    
    # Don't forget the last chunk
    if current_lines:
        chunk_text = "".join(current_lines)
        chunks.append({
            "text": chunk_text,
            "first_character_index": current_start,
            "last_character_index": current_start + current_length,
        })
    
    return chunks