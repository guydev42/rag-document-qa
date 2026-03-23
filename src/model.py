"""Retrieval models: TF-IDF, BM25, cosine similarity, re-ranking, and evaluation."""

import numpy as np
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# TF-IDF retriever
# ---------------------------------------------------------------------------

class TfidfRetriever:
    """Retrieve relevant passages using TF-IDF cosine similarity."""

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=True,
        )
        self.doc_vectors = None
        self.chunks = None
        self.metadata = None

    def fit(self, chunks: List[str], metadata: List[Dict]) -> "TfidfRetriever":
        """Index all chunks.

        Parameters
        ----------
        chunks : list of str
            Text chunks to index.
        metadata : list of dict
            Parallel metadata (doc_id, title, chunk_idx).
        """
        self.chunks = chunks
        self.metadata = metadata
        self.doc_vectors = self.vectorizer.fit_transform(chunks)
        return self

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Return top-k chunks for a query.

        Returns
        -------
        list of dict
            Each dict has: chunk, metadata, score.
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_vectors).flatten()
        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_idx:
            results.append({
                "chunk": self.chunks[idx],
                "metadata": self.metadata[idx],
                "score": float(scores[idx]),
            })
        return results

    def get_feature_names(self) -> List[str]:
        """Return vocabulary terms."""
        return self.vectorizer.get_feature_names_out().tolist()


# ---------------------------------------------------------------------------
# BM25 retriever
# ---------------------------------------------------------------------------

class BM25Retriever:
    """Retrieve relevant passages using Okapi BM25 scoring."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.chunks = None
        self.metadata = None

    def fit(self, chunks: List[str], metadata: List[Dict]) -> "BM25Retriever":
        """Index all chunks with BM25."""
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.metadata = metadata
        tokenized = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        return self

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """Return top-k chunks for a query."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_idx = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_idx:
            results.append({
                "chunk": self.chunks[idx],
                "metadata": self.metadata[idx],
                "score": float(scores[idx]),
            })
        return results


# ---------------------------------------------------------------------------
# Cross-encoder style re-ranker (no external model -- uses term overlap)
# ---------------------------------------------------------------------------

class TermOverlapReranker:
    """Re-rank retrieved passages using term overlap scoring.

    This is a lightweight stand-in for a cross-encoder model. It scores
    each query-passage pair based on:
    - Exact term overlap ratio
    - Bigram overlap bonus
    - Passage length penalty (prefer concise passages)
    """

    def __init__(self, bigram_weight: float = 0.3, length_penalty: float = 0.1):
        self.bigram_weight = bigram_weight
        self.length_penalty = length_penalty

    def score(self, query: str, passage: str) -> float:
        """Compute re-ranking score for a query-passage pair."""
        q_tokens = set(query.lower().split())
        p_tokens = set(passage.lower().split())

        if not q_tokens:
            return 0.0

        # Unigram overlap
        overlap = len(q_tokens & p_tokens) / len(q_tokens)

        # Bigram overlap
        q_bigrams = set(zip(query.lower().split(), query.lower().split()[1:]))
        p_bigrams = set(zip(passage.lower().split(), passage.lower().split()[1:]))
        bigram_overlap = 0.0
        if q_bigrams:
            bigram_overlap = len(q_bigrams & p_bigrams) / len(q_bigrams)

        # Length penalty: penalize very long passages
        length_factor = 1.0 / (1.0 + self.length_penalty * (len(p_tokens) / 100.0))

        return (overlap + self.bigram_weight * bigram_overlap) * length_factor

    def rerank(self, query: str, results: List[Dict], k: int = 5) -> List[Dict]:
        """Re-rank results using cross-encoder style scoring.

        Parameters
        ----------
        query : str
            Search query.
        results : list of dict
            Initial retrieval results with 'chunk', 'metadata', 'score'.
        k : int
            Number of results to return.

        Returns
        -------
        list of dict
            Re-ranked results with updated scores.
        """
        for r in results:
            rerank_score = self.score(query, r["chunk"])
            r["rerank_score"] = rerank_score
            # Combine original retrieval score with re-ranking score
            r["combined_score"] = 0.6 * r["score"] + 0.4 * rerank_score

        results.sort(key=lambda x: x["combined_score"], reverse=True)
        return results[:k]


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def precision_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """Compute precision@k.

    Parameters
    ----------
    retrieved_doc_ids : list of str
        Document IDs of retrieved results (ordered by rank).
    relevant_doc_ids : list of str
        Ground truth relevant document IDs.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        Precision@k value.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / k


def recall_at_k(retrieved_doc_ids: List[str], relevant_doc_ids: List[str], k: int) -> float:
    """Compute recall@k.

    Parameters
    ----------
    retrieved_doc_ids : list of str
        Document IDs of retrieved results (ordered by rank).
    relevant_doc_ids : list of str
        Ground truth relevant document IDs.
    k : int
        Cutoff rank.

    Returns
    -------
    float
        Recall@k value.
    """
    if not relevant_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    relevant_set = set(relevant_doc_ids)
    hits = sum(1 for doc_id in top_k if doc_id in relevant_set)
    return hits / len(relevant_set)


def reciprocal_rank(retrieved_doc_ids: List[str], relevant_doc_ids: List[str]) -> float:
    """Compute reciprocal rank (1/rank of first relevant result).

    Parameters
    ----------
    retrieved_doc_ids : list of str
        Document IDs of retrieved results (ordered by rank).
    relevant_doc_ids : list of str
        Ground truth relevant document IDs.

    Returns
    -------
    float
        Reciprocal rank (0 if no relevant document found).
    """
    relevant_set = set(relevant_doc_ids)
    for i, doc_id in enumerate(retrieved_doc_ids):
        if doc_id in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def mean_reciprocal_rank(
    all_retrieved: List[List[str]],
    all_relevant: List[List[str]],
) -> float:
    """Compute mean reciprocal rank across all queries.

    Parameters
    ----------
    all_retrieved : list of list of str
        Retrieved doc IDs for each query.
    all_relevant : list of list of str
        Relevant doc IDs for each query.

    Returns
    -------
    float
        MRR score.
    """
    if not all_retrieved:
        return 0.0
    rr_scores = [
        reciprocal_rank(ret, rel)
        for ret, rel in zip(all_retrieved, all_relevant)
    ]
    return float(np.mean(rr_scores))


def evaluate_retriever(
    retriever,
    eval_qa: List[Dict],
    chunks: List[str],
    metadata: List[Dict],
    k_values: List[int] = None,
) -> Dict:
    """Run full evaluation of a retriever on the Q&A set.

    Parameters
    ----------
    retriever : TfidfRetriever or BM25Retriever
        Fitted retriever.
    eval_qa : list of dict
        Q&A pairs with question, answer, relevant_doc_ids.
    chunks : list of str
        All text chunks.
    metadata : list of dict
        Parallel metadata.
    k_values : list of int
        Values of k to evaluate.

    Returns
    -------
    dict
        Evaluation metrics including precision@k, recall@k, MRR.
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    all_retrieved = []
    all_relevant = []
    per_query_results = []

    for qa in eval_qa:
        results = retriever.retrieve(qa["question"], k=max(k_values))
        retrieved_ids = [r["metadata"]["doc_id"] for r in results]
        # Deduplicate while preserving order
        seen = set()
        unique_ids = []
        for did in retrieved_ids:
            if did not in seen:
                seen.add(did)
                unique_ids.append(did)

        all_retrieved.append(unique_ids)
        all_relevant.append(qa["relevant_doc_ids"])

        per_query_results.append({
            "question": qa["question"],
            "relevant_doc_ids": qa["relevant_doc_ids"],
            "retrieved_doc_ids": unique_ids,
            "top_score": results[0]["score"] if results else 0.0,
        })

    metrics = {"mrr": mean_reciprocal_rank(all_retrieved, all_relevant)}
    for k in k_values:
        p_scores = [
            precision_at_k(ret, rel, k)
            for ret, rel in zip(all_retrieved, all_relevant)
        ]
        r_scores = [
            recall_at_k(ret, rel, k)
            for ret, rel in zip(all_retrieved, all_relevant)
        ]
        metrics[f"precision@{k}"] = float(np.mean(p_scores))
        metrics[f"recall@{k}"] = float(np.mean(r_scores))

    metrics["per_query"] = per_query_results
    return metrics
