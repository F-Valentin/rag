"""Test script for the RAG pipeline."""

import os
import sys

# Ensure src/ is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from student.chunking import chunk_markdown, chunk_python
from student.indexing import build_index
from student.retrieval import load_index, search


def test_chunking():
    """Test chunking on sample content."""
    print("=" * 50)
    print("TEST: Chunking")
    print("=" * 50)
    
    # Test markdown
    md_text = """# Introduction

This is the intro paragraph.

## Section 1

Some content here.
More content.

### Subsection

Even more content.

## Section 2

Final section.
"""
    
    md_chunks = chunk_markdown(md_text, max_chunk_size=50)
    print(f"\nMarkdown chunks: {len(md_chunks)}")
    for i, c in enumerate(md_chunks[:3]):
        print(f"  Chunk {i}: [{c['first_character_index']}:{c['last_character_index']}] "
              f"{repr(c['text'][:40])}...")
    
    # Test Python
    py_text = """import os

def hello():
    print("Hello")

class MyClass:
    def method(self):
        return 42

async def async_func():
    pass
"""
    
    py_chunks = chunk_python(py_text, max_chunk_size=100)
    print(f"\nPython chunks: {len(py_chunks)}")
    for i, c in enumerate(py_chunks):
        print(f"  Chunk {i}: [{c['first_character_index']}:{c['last_character_index']}] "
              f"{repr(c['text'][:40])}...")
    
    print("\n✅ Chunking OK\n")


def test_indexing():
    """Test indexing on a small directory."""
    print("=" * 50)
    print("TEST: Indexing")
    print("=" * 50)
    
    # Create a temporary test directory
    test_dir = "./data/output/"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a test markdown file
    with open(os.path.join(test_dir, "readme.md"), "w") as f:
        f.write("# Test\n\nThis is a test.\n\n## Section\n\nMore content here.")
    
    # Create a test Python file
    with open(os.path.join(test_dir, "test.py"), "w") as f:
        f.write('def hello():\n    print("Hello")\n\ndef world():\n    return 42\n')
    
    # Build index
    output_dir = "./data/output"
    build_index(test_dir, output_dir, max_chunk_size=50)
    
    # Check files were created
    assert os.path.exists(os.path.join(output_dir, "chunks.json"))
    print(f"\n✅ Index created in {output_dir}")
    
    # Load and verify
    retriever, chunks = load_index(output_dir)
    print(f"Loaded {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"  {c.file_path}: [{c.first_character_index}:{c.last_character_index}] "
              f"{repr(c.text[:30])}...")
    
    print("\n✅ Indexing OK\n")
    
    # # Cleanup
    # import shutil
    # shutil.rmtree(test_dir)
    # shutil.rmtree(output_dir)


def test_retrieval():
    """Test retrieval on the temporary index."""
    print("=" * 50)
    print("TEST: Retrieval")
    print("=" * 50)
    
    # Recreate test index
    test_dir = "/tmp/test_repo"
    os.makedirs(test_dir, exist_ok=True)
    
    with open(os.path.join(test_dir, "readme.md"), "w") as f:
        f.write("# Test Project\n\nThis project handles machine learning.\n\n## Features\n\nWe support neural networks and deep learning.")
    
    with open(os.path.join(test_dir, "test.py"), "w") as f:
        f.write('def train_model():\n    """Train a neural network."""\n    pass\n\ndef evaluate():\n    return 0.95\n')
    
    output_dir = "/tmp/test_index"
    build_index(test_dir, output_dir, max_chunk_size=100)
    
    # Test search
    retriever, chunks = load_index(output_dir)
    
    queries = [
        "neural network",
        "train model",
        "evaluate performance",
    ]
    
    for q in queries:
        print(f"\nQuery: {q!r}")
        sources = search(q, retriever, chunks, k=2)
        for s in sources:
            print(f"  → {s.file_path} [{s.first_character_index}:{s.last_character_index}]")
    
    print("\n✅ Retrieval OK\n")
    
    # Cleanup
    import shutil
    shutil.rmtree(test_dir)
    shutil.rmtree(output_dir)


if __name__ == "__main__":
    test_chunking()
    test_indexing()
    test_retrieval()
    print("=" * 50)
    print("ALL TESTS PASSED!")
    print("=" * 50)