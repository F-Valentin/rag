"""Answer generation using an LLM (Qwen/Qwen3-0.6B by default) over retrieved context."""

from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING, cast

from src.models import IndexedChunk, MinimalSource

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase, PreTrainedModel

_tokenizer: Optional["PreTrainedTokenizerBase"] = None
_model: Optional["PreTrainedModel"] = None

_MODEL_NAME = "Qwen/Qwen3-0.6B"



def _load_model(model_name: str = _MODEL_NAME) -> Tuple["PreTrainedTokenizerBase", "PreTrainedModel"]:
    global _tokenizer, _model

    tokenizer = _tokenizer
    model = _model

    if model is None or tokenizer is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        _tokenizer, _model = tokenizer, model

    return tokenizer, model


def build_context_from_chunks(chunks: List[IndexedChunk], max_context_length: int = 4000) -> str:
    """Build a context string from in-memory chunks (text already available)."""
    parts: List[str] = []
    total = 0

    for chunk in chunks:
        header = f"# Source: {chunk.file_path} [{chunk.first_character_index}:{chunk.last_character_index}]\n"
        piece = header + chunk.text.strip() + "\n\n"
        if total + len(piece) > max_context_length:
            break
        parts.append(piece)
        total += len(piece)

    return "".join(parts)


def build_context_from_sources(sources: List[MinimalSource], max_context_length: int = 4000) -> str:
    """Build a context string by re-reading source text from disk.

    Used when only MinimalSource references (file_path + character indices)
    are available, e.g. when re-hydrating saved search results.
    """
    parts: List[str] = []
    total = 0

    for source in sources:
        try:
            with open(source.file_path, "r", encoding="utf-8") as f:
                content = f.read()
            text = content[source.first_character_index:source.last_character_index]
        except (OSError, UnicodeDecodeError):
            continue

        header = f"# Source: {source.file_path} [{source.first_character_index}:{source.last_character_index}]\n"
        piece = header + text.strip() + "\n\n"
        if total + len(piece) > max_context_length:
            break
        parts.append(piece)
        total += len(piece)

    return "".join(parts)


def _prompt(question: str, context: str) -> str:
    return (
        "You are a helpful assistant answering questions about a codebase using "
        "only the information provided in the context below. Be self-contained, "
        "cite the source file(s) you used, and do not invent information that is "
        "not present in the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def generate_answer(
    question: str,
    context: str,
    model_name: str = _MODEL_NAME,
    max_new_tokens: int = 800,
) -> str:
    """Generate a natural-language answer given a question and a context string."""
    try:
        tokenizer, model = _load_model(model_name)

        prompt = _prompt(question, context)
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(cast(str, text), return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        generated = outputs[0][inputs["input_ids"].shape[-1]:]
        answer = cast(str, tokenizer.decode(generated, skip_special_tokens=True))
        return answer.strip()
    except Exception as exc:  # noqa: BLE001 - surface a graceful fallback instead of crashing
        return f"[generation failed: {exc}]"
