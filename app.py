"""Streamlit dashboard for RAG document question answering."""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

PROJECT_DIR = os.path.dirname(__file__)
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import load_documents, build_chunk_index, load_eval_qa, preprocess_text
from src.model import (
    TfidfRetriever, BM25Retriever, TermOverlapReranker,
    evaluate_retriever, precision_at_k, recall_at_k, mean_reciprocal_rank,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG document Q&A", layout="wide")

GOLD = "#E8C230"
NAVY = "#3B6FD4"
DARK_BG = "#162240"


@st.cache_data
def load_data():
    docs_path = os.path.join(PROJECT_DIR, "data", "documents.json")
    qa_path = os.path.join(PROJECT_DIR, "data", "eval_qa.json")
    documents = load_documents(docs_path)
    eval_qa = load_eval_qa(qa_path)
    return documents, eval_qa


@st.cache_resource
def build_retrievers(chunk_size, overlap):
    documents, _ = load_data()
    chunks, metadata = build_chunk_index(documents, chunk_size=chunk_size, overlap=overlap)
    tfidf = TfidfRetriever().fit(chunks, metadata)
    bm25 = BM25Retriever().fit(chunks, metadata)
    return chunks, metadata, tfidf, bm25


def highlight_terms(text, query):
    """Highlight matching query terms in retrieved text."""
    query_terms = set(query.lower().split())
    words = text.split()
    highlighted = []
    for word in words:
        clean = re.sub(r"[^a-z0-9]", "", word.lower())
        if clean in query_terms:
            highlighted.append(f"**:orange[{word}]**")
        else:
            highlighted.append(word)
    return " ".join(highlighted)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["Query", "Documents", "Evaluation", "Metrics dashboard"],
)

chunk_size = st.sidebar.slider("Chunk size", 200, 1000, 500, step=50)
overlap = st.sidebar.slider("Chunk overlap", 0, 200, 50, step=10)
top_k = st.sidebar.slider("Top-k results", 1, 15, 5)
retriever_type = st.sidebar.selectbox("Retriever", ["TF-IDF", "BM25"])
use_reranker = st.sidebar.checkbox("Apply re-ranking", value=True)

documents, eval_qa = load_data()
chunks, metadata, tfidf_retriever, bm25_retriever = build_retrievers(chunk_size, overlap)
retriever = tfidf_retriever if retriever_type == "TF-IDF" else bm25_retriever
reranker = TermOverlapReranker()


# ---------------------------------------------------------------------------
# Page: Query
# ---------------------------------------------------------------------------
if page == "Query":
    st.title("RAG document question answering")
    st.markdown("Ask questions about Calgary municipal policies and retrieve relevant passages.")

    query = st.text_input(
        "Enter your question",
        placeholder="e.g., What is Calgary's affordable housing target?",
    )

    if query:
        results = retriever.retrieve(query, k=top_k * 2 if use_reranker else top_k)
        if use_reranker:
            results = reranker.rerank(query, results, k=top_k)

        st.markdown(f"### Top {len(results)} retrieved passages")
        st.markdown(f"**Retriever:** {retriever_type} | **Re-ranking:** {'Yes' if use_reranker else 'No'}")

        for i, r in enumerate(results):
            score_key = "combined_score" if use_reranker and "combined_score" in r else "score"
            score = r[score_key]
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"**Rank {i+1}** — {r['metadata']['title']} (chunk {r['metadata']['chunk_idx']})")
                st.markdown(highlight_terms(r["chunk"], query))
            with col2:
                st.metric("Score", f"{score:.4f}")
            st.divider()

    # Quick examples
    st.markdown("### Example questions")
    examples = [
        "What zoning districts are available for residential use?",
        "How does Calgary handle emergency flood response?",
        "What is the waste diversion rate in Calgary?",
        "How much does a business licence cost?",
        "What is the property tax rate for residential properties?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["query"] = ex
            st.rerun()


# ---------------------------------------------------------------------------
# Page: Documents
# ---------------------------------------------------------------------------
elif page == "Documents":
    st.title("Document corpus")
    st.markdown(f"**{len(documents)} documents** | **{len(chunks)} chunks** (chunk_size={chunk_size}, overlap={overlap})")

    # Document overview table
    doc_data = []
    for doc in documents:
        doc_chunks = [m for m in metadata if m["doc_id"] == doc["doc_id"]]
        doc_data.append({
            "Document ID": doc["doc_id"],
            "Title": doc["title"],
            "Characters": len(doc["text"]),
            "Words": len(doc["text"].split()),
            "Chunks": len(doc_chunks),
        })
    df_docs = pd.DataFrame(doc_data)
    st.dataframe(df_docs, use_container_width=True, hide_index=True)

    # Chunk length distribution
    chunk_lengths = [len(c) for c in chunks]
    fig = px.histogram(
        x=chunk_lengths,
        nbins=30,
        title="Chunk length distribution (characters)",
        labels={"x": "Characters", "y": "Count"},
        color_discrete_sequence=[NAVY],
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Document viewer
    selected_doc = st.selectbox(
        "View document",
        [f"{d['doc_id']}: {d['title']}" for d in documents],
    )
    doc_id = selected_doc.split(":")[0]
    doc = next(d for d in documents if d["doc_id"] == doc_id)
    st.markdown(f"### {doc['title']}")
    st.text(doc["text"])


# ---------------------------------------------------------------------------
# Page: Evaluation
# ---------------------------------------------------------------------------
elif page == "Evaluation":
    st.title("Retrieval evaluation")
    st.markdown(f"Evaluating {retriever_type} retriever on {len(eval_qa)} Q&A pairs.")

    metrics = evaluate_retriever(retriever, eval_qa, chunks, metadata, k_values=[1, 3, 5, 10])

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MRR", f"{metrics['mrr']:.3f}")
    col2.metric("Precision@1", f"{metrics['precision@1']:.3f}")
    col3.metric("Precision@3", f"{metrics['precision@3']:.3f}")
    col4.metric("Recall@5", f"{metrics['recall@5']:.3f}")

    # Per-k metrics chart
    k_data = []
    for k in [1, 3, 5, 10]:
        k_data.append({"k": k, "Metric": "Precision", "Value": metrics[f"precision@{k}"]})
        k_data.append({"k": k, "Metric": "Recall", "Value": metrics[f"recall@{k}"]})
    df_k = pd.DataFrame(k_data)

    fig = px.bar(
        df_k, x="k", y="Value", color="Metric", barmode="group",
        title=f"Precision and recall at different k values ({retriever_type})",
        color_discrete_map={"Precision": GOLD, "Recall": NAVY},
    )
    fig.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    # Per-query details
    st.markdown("### Per-query results")
    pq_data = []
    for pq in metrics["per_query"]:
        rr = 0.0
        for i, did in enumerate(pq["retrieved_doc_ids"]):
            if did in pq["relevant_doc_ids"]:
                rr = 1.0 / (i + 1)
                break
        pq_data.append({
            "Question": pq["question"][:80],
            "Relevant": ", ".join(pq["relevant_doc_ids"]),
            "Top retrieved": ", ".join(pq["retrieved_doc_ids"][:3]),
            "RR": f"{rr:.2f}",
            "Top score": f"{pq['top_score']:.4f}",
        })
    st.dataframe(pd.DataFrame(pq_data), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Metrics dashboard
# ---------------------------------------------------------------------------
elif page == "Metrics dashboard":
    st.title("Retrieval metrics dashboard")
    st.markdown("Compare TF-IDF and BM25 retrievers across evaluation metrics.")

    # Evaluate both retrievers
    tfidf_metrics = evaluate_retriever(tfidf_retriever, eval_qa, chunks, metadata)
    bm25_metrics = evaluate_retriever(bm25_retriever, eval_qa, chunks, metadata)

    # Side by side comparison
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### TF-IDF")
        st.metric("MRR", f"{tfidf_metrics['mrr']:.3f}")
        st.metric("Precision@3", f"{tfidf_metrics['precision@3']:.3f}")
        st.metric("Recall@5", f"{tfidf_metrics['recall@5']:.3f}")
    with col2:
        st.markdown("### BM25")
        st.metric("MRR", f"{bm25_metrics['mrr']:.3f}")
        st.metric("Precision@3", f"{bm25_metrics['precision@3']:.3f}")
        st.metric("Recall@5", f"{bm25_metrics['recall@5']:.3f}")

    # Comparison bar chart
    comparison_data = []
    for k in [1, 3, 5, 10]:
        comparison_data.append({"k": k, "Retriever": "TF-IDF", "Precision": tfidf_metrics[f"precision@{k}"]})
        comparison_data.append({"k": k, "Retriever": "BM25", "Precision": bm25_metrics[f"precision@{k}"]})
    df_comp = pd.DataFrame(comparison_data)

    fig = px.bar(
        df_comp, x="k", y="Precision", color="Retriever", barmode="group",
        title="Precision@k comparison: TF-IDF vs BM25",
        color_discrete_map={"TF-IDF": GOLD, "BM25": NAVY},
    )
    fig.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    # MRR comparison across chunk sizes
    st.markdown("### MRR sensitivity to chunk size")
    mrr_data = []
    for cs in [200, 300, 400, 500, 600, 800, 1000]:
        cs_chunks, cs_meta = build_chunk_index(documents, chunk_size=cs, overlap=50)
        tfidf_cs = TfidfRetriever().fit(cs_chunks, cs_meta)
        bm25_cs = BM25Retriever().fit(cs_chunks, cs_meta)
        tfidf_m = evaluate_retriever(tfidf_cs, eval_qa, cs_chunks, cs_meta)
        bm25_m = evaluate_retriever(bm25_cs, eval_qa, cs_chunks, cs_meta)
        mrr_data.append({"Chunk size": cs, "Retriever": "TF-IDF", "MRR": tfidf_m["mrr"]})
        mrr_data.append({"Chunk size": cs, "Retriever": "BM25", "MRR": bm25_m["mrr"]})

    df_mrr = pd.DataFrame(mrr_data)
    fig = px.line(
        df_mrr, x="Chunk size", y="MRR", color="Retriever",
        title="MRR vs chunk size",
        color_discrete_map={"TF-IDF": GOLD, "BM25": NAVY},
        markers=True,
    )
    fig.update_layout(yaxis_range=[0, 1.05])
    st.plotly_chart(fig, use_container_width=True)

    # Score distribution
    st.markdown("### Score distributions")
    tfidf_scores = []
    bm25_scores = []
    for qa in eval_qa:
        tr = tfidf_retriever.retrieve(qa["question"], k=5)
        br = bm25_retriever.retrieve(qa["question"], k=5)
        tfidf_scores.extend([r["score"] for r in tr])
        bm25_scores.extend([r["score"] for r in br])

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=tfidf_scores, name="TF-IDF", marker_color=GOLD, opacity=0.7, nbinsx=30))
    fig.add_trace(go.Histogram(x=bm25_scores, name="BM25", marker_color=NAVY, opacity=0.7, nbinsx=30))
    fig.update_layout(
        title="Retrieval score distributions",
        xaxis_title="Score",
        yaxis_title="Count",
        barmode="overlay",
    )
    st.plotly_chart(fig, use_container_width=True)
