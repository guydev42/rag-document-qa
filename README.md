<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=RAG%20Document%20Q%26A&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Retrieval-augmented%20generation%20for%20municipal%20policy%20question%20answering&descAlignY=55&descSize=16" width="100%"/>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Documents-15-9558B2?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/MRR-0.82-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Precision@3-0.89-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Active-22c55e?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge"/>
</p>

<p>
  <a href="#overview">Overview</a> •
  <a href="#key-results">Key results</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#methodology">Methodology</a>
</p>

</div>

---

## Overview

> **A self-contained retrieval-augmented generation system that answers questions about municipal policy documents using TF-IDF and BM25 retrieval with re-ranking.**

When organizations accumulate policy documents, bylaws, and operational guides, finding the right passage to answer a specific question becomes slow and error-prone. This project builds a RAG pipeline that indexes municipal policy documents into overlapping text chunks, retrieves the most relevant passages for a given question, and presents them with relevance scores and highlighted matching terms. No external API calls are needed -- all retrieval and scoring runs locally using scikit-learn and rank_bm25.

```
Problem   →  Finding answers in a growing corpus of municipal policy documents
Solution  →  TF-IDF and BM25 retrieval with term-overlap re-ranking
Impact    →  MRR 0.82, Precision@3 0.89 across 30 evaluation questions on 15 documents
```

---

## Key results

| Metric | TF-IDF | BM25 |
|--------|--------|------|
| MRR | 0.82 | 0.80 |
| Precision@1 | 0.77 | 0.73 |
| Precision@3 | 0.89 | 0.87 |
| Recall@5 | 0.93 | 0.90 |

---

## Architecture

```
┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Document        │───▶│  Text chunking   │───▶│  TF-IDF / BM25   │
│  loading         │    │  with overlap     │    │  indexing         │
└──────────────────┘    └──────────────────┘    └────────┬─────────┘
                                                         │
                          ┌──────────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Cosine similarity   │───▶│  Term-overlap        │
              │  retrieval           │    │  re-ranking          │
              └──────────────────────┘    └──────────┬───────────┘
                                                     │
                          ┌──────────────────────────┘
                          ▼
              ┌──────────────────────┐    ┌──────────────────────┐
              │  Passage ranking     │───▶│  Answer              │
              │  with scores         │    │  presentation        │
              └──────────────────────┘    └──────────────────────┘
```

<details>
<summary><b>Project structure</b></summary>

```
project_18_rag_document_qa/
├── data/
│   ├── documents.json                 # 15 municipal policy documents
│   ├── eval_qa.json                   # 30 evaluation Q&A pairs
│   └── generate_data.py               # Synthetic data generator
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Document loading and chunking
│   └── model.py                       # Retrieval models and evaluation
├── notebooks/
│   ├── 01_eda.ipynb                   # Document statistics and vocabulary
│   ├── 02_feature_engineering.ipynb   # Text preprocessing and indexing
│   ├── 03_modeling.ipynb              # TF-IDF vs BM25 comparison
│   └── 04_evaluation.ipynb            # Full evaluation and error analysis
├── figures/
├── app.py                             # Streamlit dashboard
├── requirements.txt
└── README.md
```

</details>

---

## Quickstart

```bash
# Clone and navigate
git clone https://github.com/guydev42/calgary-data-portfolio.git
cd calgary-data-portfolio/project_18_rag_document_qa

# Install dependencies
pip install -r requirements.txt

# Generate document data
python data/generate_data.py

# Launch dashboard
streamlit run app.py
```

---

## Dataset

| Property | Details |
|----------|---------|
| Source | Synthetic municipal policy documents |
| Documents | 15 (land use, transit, water, housing, parks, etc.) |
| Evaluation questions | 30 with ground truth document IDs |
| Chunk size | 500 characters with 50-character overlap |
| Domain | Calgary municipal policy and public services |

---

## Tech stack

<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white"/>
  <img src="https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white"/>
</p>

---

## Methodology

<details>
<summary><b>Document chunking</b></summary>

- Fixed-size character chunks (default 500 characters) with configurable overlap (default 50)
- Sentence boundary detection to avoid splitting mid-sentence
- Each chunk retains metadata linking it back to the source document
</details>

<details>
<summary><b>TF-IDF retrieval</b></summary>

- Scikit-learn TfidfVectorizer with sublinear TF scaling and English stop words
- Unigram and bigram features up to 5,000 terms
- Cosine similarity between query vector and all chunk vectors
</details>

<details>
<summary><b>BM25 retrieval</b></summary>

- Okapi BM25 with k1=1.5 and b=0.75 parameters
- Token-level matching with term frequency saturation
- Length normalization relative to average document length
</details>

<details>
<summary><b>Re-ranking</b></summary>

- Term overlap scoring as a lightweight cross-encoder alternative
- Combines unigram overlap, bigram overlap bonus, and passage length penalty
- Weighted combination: 60% retrieval score + 40% re-ranking score
</details>

<details>
<summary><b>Evaluation</b></summary>

- 30 hand-crafted questions with ground truth relevant document IDs
- Metrics: Precision@k, Recall@k, MRR (mean reciprocal rank)
- Parameter sensitivity analysis across chunk sizes and k values
</details>

---

## Acknowledgements

Built as part of the [Calgary Data Portfolio](https://guydev42.github.io/calgary-data-portfolio/).

---

<div align="center">
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**[Ola K.](https://github.com/guydev42)**
</div>
