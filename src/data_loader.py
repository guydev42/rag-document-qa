"""Load, chunk, and prepare text documents for retrieval."""

import json
import os
import re
from typing import List, Dict, Tuple


def load_documents(path: str) -> List[Dict]:
    """Load documents from a JSON file.

    Parameters
    ----------
    path : str
        Path to documents.json.

    Returns
    -------
    list of dict
        Each dict has keys: doc_id, title, text.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
) -> List[str]:
    """Split text into overlapping chunks by character count.

    Parameters
    ----------
    text : str
        Source text to split.
    chunk_size : int
        Maximum characters per chunk.
    overlap : int
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    list of str
        Text chunks.
    """
    if not text or chunk_size <= 0:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # Try to break at a sentence boundary within the last 20% of the chunk
        if end < len(text):
            boundary = chunk.rfind(". ", int(chunk_size * 0.8))
            if boundary != -1:
                chunk = chunk[: boundary + 1]
                end = start + boundary + 1
        chunks.append(chunk.strip())
        start = end - overlap
    return [c for c in chunks if c]


def build_chunk_index(
    documents: List[Dict],
    chunk_size: int = 500,
    overlap: int = 50,
) -> Tuple[List[str], List[Dict]]:
    """Chunk all documents and build a flat index.

    Returns
    -------
    chunks : list of str
        All text chunks across documents.
    metadata : list of dict
        Parallel list with doc_id, title, chunk_idx for each chunk.
    """
    chunks = []
    metadata = []
    for doc in documents:
        doc_chunks = chunk_text(doc["text"], chunk_size, overlap)
        for idx, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            metadata.append({
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "chunk_idx": idx,
            })
    return chunks, metadata


def load_eval_qa(path: str) -> List[Dict]:
    """Load evaluation question-answer pairs.

    Parameters
    ----------
    path : str
        Path to eval_qa.json.

    Returns
    -------
    list of dict
        Each dict has keys: question, answer, relevant_doc_ids.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def preprocess_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip extra whitespace, remove special chars."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s.,;:!?'-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def generate_synthetic_qa(
    documents: List[Dict],
    n_questions: int = 30,
) -> List[Dict]:
    """Generate simple synthetic Q&A pairs from documents.

    Extracts key sentences from each document and turns them into questions
    by identifying the main topic. This is a heuristic approach for
    evaluation without requiring an LLM.

    Parameters
    ----------
    documents : list of dict
        Source documents with doc_id, title, text.
    n_questions : int
        Target number of questions.

    Returns
    -------
    list of dict
        Q&A pairs with question, answer, relevant_doc_ids.
    """
    qa_pairs = []
    questions_per_doc = max(1, n_questions // len(documents))

    for doc in documents:
        sentences = [s.strip() for s in doc["text"].split(".") if len(s.strip()) > 30]
        selected = sentences[:questions_per_doc]
        for sent in selected:
            words = sent.split()
            # Create a question from the first meaningful clause
            if len(words) > 5:
                qa_pairs.append({
                    "question": f"What does the policy say about {' '.join(words[2:8]).lower().strip('.,;:')}?",
                    "answer": sent.strip() + ".",
                    "relevant_doc_ids": [doc["doc_id"]],
                })

    return qa_pairs[:n_questions]
