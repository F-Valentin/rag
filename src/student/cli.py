"""CLI interface for the RAG system."""

def main() -> dict:
    """Expose commands to python-fire."""
    return {
        "hello": hello_command,
    }

def hello_command(name: str = "world") -> None:
    """Say hello. Test command."""
    print(f"Hello, {name}!")