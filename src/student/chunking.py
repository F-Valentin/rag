import re
import ast

def chunk_markdown(text: str, max_chunk_size: int) -> list[dict]:
    pattern = re.compile(r'(?=^#{1,6} )', re.MULTILINE)
    sections = pattern.split(text.strip())
    chunks = []
    start = 0

    for section in sections:
        if len(section) <= max_chunk_size:
            chunks.append({
                "text": section,
                "first_character_index": start,
                "last_character_index": start + len(section)
            })
        else:
            for i in range(0, len(section), max_chunk_size):
                chunk = section[i:i + max_chunk_size]
                chunks.append({
                    "text": chunk,
                    "first_character_index": start + i,
                    "last_character_index": start + i + len(chunk)
                })
        start += len(section)
    return chunks


def chunk_python(source: str, max_chunk_size: int) -> list[dict]:
    tree = ast.parse(source)
    chunks = []

    top_level = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]

    for node in top_level:
        start_char = _line_to_char(source, node.lineno)
        end_char = _line_to_char(source, node.end_lineno + 1)
        text = source[start_char:end_char]

        if len(text) <= max_chunk_size:
            chunks.append({
                "text": text,
                "first_character_index": start_char,
                "last_character_index": end_char
            })
        else:
            for i in range(0, len(text), max_chunk_size):
                chunk = text[i:i + max_chunk_size]
                chunks.append({
                    "text": chunk,
                    "first_character_index": start_char + i,
                    "last_character_index": start_char + i + len(chunk)
                })
    return chunks

def _line_to_char(source: str, lineno: int) -> int:
    lines = source.splitlines(keepends=True)
    return sum(len(l) for l in lines[:lineno - 1])