# Multi-Doc Hybrid Search & Reranking Engine

A production-ready Hybrid Search Engine integrating Dense Vector Search, Sparse Keyword Search (BM25), Reciprocal Rank Fusion (RRF), and Cross-Encoder Reranking to deliver state-of-the-art document retrieval.

## Key Features

- **Hybrid Retrieval**: Combines semantic meaning (Dense Vectors) with exact keyword matching (Sparse BM25).
- **Advanced Reranking**: Utilizes `cross-encoder/ms-marco-MiniLM-L-6-v2` cross-encoder for lightning-fast precision on top candidates.
- **Rich User Interface**: Features a glassmorphic frontend UI for instantly uploading files (PDF, TXT, etc.) and querying the engine.
- **Production Hardened**: Features embedding caching (`lru_cache`), and cross-encoder graceful degradation (timeout fallback).
- **Benchmarking Suite**: Included asynchronous load-testing script to monitor multi-stage latencies (Avg, Median, P95, P99).

## Tech Stack

- **API Framework**: FastAPI, Uvicorn, Pydantic
- **Vector Database**: Qdrant (`qdrant/qdrant`)
- **Machine Learning**: `sentence-transformers`, `rank_bm25`, `pypdf`
- **Models**:
  - Dense: `all-MiniLM-L6-v2`
  - Cross-Encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

## 🚀 Getting Started

### 1. Launch the Server Locally

Ensure you have Python installed, then run:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server natively
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- The API will be available at `http://localhost:8000`
- API Documentation (Swagger) at `http://localhost:8000/docs`
- Qdrant runs purely in-memory by default.

### 2. Run Benchmarks

To load-test the API and evaluate pipeline latencies (1,000 synthetic documents, 100 sequential queries, 50 concurrent queries):

```bash
# Requires httpx and asyncio
pip install httpx
python scripts/benchmark.py
```

Results are printed to the console as a Markdown table and exported to `results/benchmark_report.json`.

---

## 📡 API Endpoints

### `POST /ingest`
Indexes a document by chunking it and computing dense/sparse representations.
**Payload:**
```json
{
  "document_id": "doc-1",
  "text": "The quick brown fox...",
  "metadata": {"category": "test"}
}
```

### `POST /upload`
Uploads files (`.txt`, `.pdf`, `.md`, `.csv`, `.json`) and ingests their text seamlessly into the engine.

### `POST /search`
Performs keyword, vector, or hybrid search.
**Payload:**
```json
{
  "query": "quick fox",
  "search_mode": "hybrid",
  "top_k": 5
}
```

### `GET /health`
Returns the status of the API, Qdrant connection, BM25 index, and chunk count.
