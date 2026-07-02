"""Pydantic models for the RAG pipeline."""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A minimal source of information with file path and character indices."""
    file_path: str
    first_character_index: int
    last_character_index: int


class IndexedChunk(BaseModel):
    """A chunk of text indexed for BM25 retrieval.
    
    Internal use only — contains the text for indexing/searching.
    """
    file_path: str
    text: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without known answer or sources."""
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with ground-truth sources and answer."""
    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions."""
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Search results for a single question."""
    question_id: str
    question_str: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results with a generated answer."""
    answer: str


class StudentSearchResults(BaseModel):
    """Collection of search results for a dataset."""
    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(StudentSearchResults):
    """Collection of search results with generated answers."""
    search_results: list[MinimalAnswer]