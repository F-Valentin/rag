"""CLI interface for the RAG system."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from tqdm import tqdm

from src.evaluation import evaluate as evaluate_results
from src.generation import (
    build_context_from_chunks,
    build_context_from_sources,
    generate_answer,
)
from src.indexing import build_index
from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from src.retrieval import load_index, search as run_search, search_full

_DEFAULT_INDEX_DIR = "data/processed"
_DEFAULT_MAX_CHUNK_SIZE = 2000
_DEFAULT_K = 10
_DEFAULT_MAX_CONTEXT_LENGTH = 4000


def main() -> dict:
    """Expose commands to python-fire."""
    return {
        "hello": hello_command,
        "index": index_command,
        "search": search_command,
        "search_dataset": search_dataset_command,
        "answer": answer_command,
        "answer_dataset": answer_dataset_command,
        "evaluate": evaluate_command,
    }


def hello_command(name: str = "world") -> None:
    """Say hello. Test command."""
    print(f"Hello, {name}!")


def _load_rag_dataset(dataset_path: str) -> RagDataset:
    """Load and validate a RagDataset from a JSON file on disk."""
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return RagDataset(**data)


def _save_json_model(model: Any, output_path: str) -> None:
    """Save a pydantic model as pretty-printed JSON, creating parent dirs."""
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(model.model_dump(), f, indent=2)


def index_command(
    repo_path: str = "./data/raw/vllm-0.10.1",
    output_dir: str = _DEFAULT_INDEX_DIR,
    max_chunk_size: int = _DEFAULT_MAX_CHUNK_SIZE,
) -> None:
    """Index a repository, building a searchable BM25 knowledge base.

    Args:
        repo_path: Path to the repository root to ingest.
        output_dir: Directory where the index and chunks will be saved.
        max_chunk_size: Maximum characters per chunk.
    """
    try:
        build_index(repo_path, output_dir, max_chunk_size)
        print(f"Ingestion complete! Indices saved under {output_dir}/")
    except (OSError, ValueError) as exc:
        print(f"Indexing failed: {exc}")


def search_command(
    query: str,
    index_dir: str = _DEFAULT_INDEX_DIR,
    k: int = _DEFAULT_K,
) -> None:
    """Search the index for a single query and print the results as JSON.

    Args:
        query: The natural-language query to search for.
        index_dir: Directory containing the saved index and chunks.
        k: Number of results to retrieve.
    """
    try:
        retriever, chunks = load_index(index_dir)
    except (OSError, FileNotFoundError) as exc:
        print(f"Could not load index from {index_dir!r}: {exc}")
        return

    if not query.strip():
        print("Query is empty; nothing to search for.")
        return

    sources = run_search(query, retriever, chunks, k)
    print(json.dumps([s.model_dump() for s in sources], indent=2))


def search_dataset_command(
    dataset_path: str,
    index_dir: str = _DEFAULT_INDEX_DIR,
    k: int = _DEFAULT_K,
    save_directory: str = "data/output/search_results",
) -> None:
    """Run retrieval for every question in a dataset and save the results.

    Args:
        dataset_path: Path to a JSON file following the RagDataset model.
        index_dir: Directory containing the saved index and chunks.
        k: Number of results to retrieve per question.
        save_directory: Directory where results will be saved (filename
            matches the input dataset's basename).
    """
    try:
        dataset = _load_rag_dataset(dataset_path)
    except (OSError, ValueError) as exc:
        print(f"Could not load dataset from {dataset_path!r}: {exc}")
        return

    try:
        retriever, chunks = load_index(index_dir)
    except (OSError, FileNotFoundError) as exc:
        print(f"Could not load index from {index_dir!r}: {exc}")
        return

    results: List[MinimalSearchResults] = []
    for question in tqdm(dataset.rag_questions, desc="Searching dataset"):
        sources = run_search(question.question, retriever, chunks, k)
        results.append(
            MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources,
            )
        )

    student_results = StudentSearchResults(search_results=results, k=k)

    output_path = os.path.join(save_directory, os.path.basename(dataset_path))
    _save_json_model(student_results, output_path)
    print(f"Saved student_search_results to {output_path}")


def answer_command(
    question: str,
    index_dir: str = _DEFAULT_INDEX_DIR,
    k: int = _DEFAULT_K,
    max_context_length: int = _DEFAULT_MAX_CONTEXT_LENGTH,
) -> None:
    """Answer a single question using retrieved context from the index.

    Args:
        question: The natural-language question to answer.
        index_dir: Directory containing the saved index and chunks.
        k: Number of chunks to retrieve for context.
        max_context_length: Maximum number of characters passed to the LLM.
    """
    try:
        retriever, chunks = load_index(index_dir)
    except (OSError, FileNotFoundError) as exc:
        print(f"Could not load index from {index_dir!r}: {exc}")
        return

    if not question.strip():
        print("Question is empty; nothing to answer.")
        return

    retrieved_chunks = search_full(question, retriever, chunks, k)
    context = build_context_from_chunks(retrieved_chunks, max_context_length)
    generated = generate_answer(question, context)

    sources = [
        MinimalSource(
            file_path=c.file_path,
            first_character_index=c.first_character_index,
            last_character_index=c.last_character_index,
        )
        for c in retrieved_chunks
    ]
    result = MinimalAnswer(
        question_id="",
        question=question,
        retrieved_sources=sources,
        answer=generated,
    )
    print(json.dumps(result.model_dump(), indent=2))


def answer_dataset_command(
    student_search_results_path: str,
    save_directory: str = "data/output/search_results_and_answer",
    max_context_length: int = _DEFAULT_MAX_CONTEXT_LENGTH,
) -> None:
    """Generate answers for every question in a saved search-results file.

    Args:
        student_search_results_path: Path to a StudentSearchResults JSON file.
        save_directory: Directory where answers will be saved (filename
            matches the input file's basename).
        max_context_length: Maximum number of characters passed to the LLM.
    """
    try:
        with open(student_search_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        student_results = StudentSearchResults(**data)
    except (OSError, ValueError) as exc:
        print(
            f"Could not load search results from "
            f"{student_search_results_path!r}: {exc}"
        )
        return

    total = len(student_results.search_results)
    print(f"Loaded {total} questions from {student_search_results_path}")

    answers: List[MinimalAnswer] = []
    for result in tqdm(
        student_results.search_results, desc="Answering questions"
    ):
        context = build_context_from_sources(
            result.retrieved_sources, max_context_length
        )
        generated = generate_answer(result.question, context)
        answers.append(
            MinimalAnswer(
                question_id=result.question_id,
                question=result.question,
                retrieved_sources=result.retrieved_sources,
                answer=generated,
            )
        )

    print(f"Processed {len(answers)} of {total} questions")

    student_results_and_answer = StudentSearchResultsAndAnswer(
        search_results=answers, k=student_results.k
    )

    output_path = os.path.join(
        save_directory, os.path.basename(student_search_results_path)
    )
    _save_json_model(student_results_and_answer, output_path)
    print(f"Saved student_search_results_and_answer to {output_path}")


def evaluate_command(
    student_answer_path: str,
    dataset_path: str,
    k: int = _DEFAULT_K,
    ks: str = "1,3,5,10",
) -> None:
    """Evaluate saved search results against ground-truth annotations.

    Args:
        student_answer_path: Path to a StudentSearchResults(-like) JSON file.
        dataset_path: Path to a RagDataset JSON file with ground-truth sources.
        k: Maximum k used for top-k truncation when comparing.
        ks: Comma-separated list of k values to report recall@k for.
    """
    try:
        with open(student_answer_path, "r", encoding="utf-8") as f:
            student_data = json.load(f)
    except (OSError, ValueError) as exc:
        print(
            f"Could not load student answers from "
            f"{student_answer_path!r}: {exc}"
        )
        return

    try:
        ground_truth_dataset = _load_rag_dataset(dataset_path)
    except (OSError, ValueError) as exc:
        print(
            f"Could not load ground-truth dataset from "
            f"{dataset_path!r}: {exc}"
        )
        return

    try:
        k_values = [int(x) for x in ks.split(",") if x.strip()]
    except ValueError:
        print(f"Invalid ks value: {ks!r}; expected comma-separated integers.")
        return

    ground_truth: Dict[str, List[MinimalSource]] = {}
    for q in ground_truth_dataset.rag_questions:
        if isinstance(q, AnsweredQuestion):
            ground_truth[q.question_id] = q.sources
        elif isinstance(q, UnansweredQuestion):
            continue

    predictions: Dict[str, List[MinimalSource]] = {}
    student_sources_count = 0
    for result in student_data.get("search_results", []):
        qid = result.get("question_id", "")
        raw_sources = result.get("retrieved_sources", [])
        sources = [MinimalSource(**s) for s in raw_sources]
        predictions[qid] = sources
        if sources:
            student_sources_count += 1

    with_sources = sum(1 for s in ground_truth.values() if s)
    print(f"Total number of questions: {len(ground_truth)}")
    print(f"Total number of questions with sources: {with_sources}")
    print(
        f"Total number of questions with student sources: "
        f"{student_sources_count}"
    )

    _ = k  # k kept for interface parity; per-question truncation uses `ks`
    results = evaluate_results(ground_truth, predictions, ks=k_values)

    print("\nEvaluation Results")
    print("=" * 40)
    print(f"Questions evaluated: {len(ground_truth)}")
    for key in sorted(results, key=lambda x: int(x.split("@")[1])):
        print(f"{key.capitalize()}: {results[key]:.3f}")
