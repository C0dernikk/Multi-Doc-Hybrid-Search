# Multi-Doc Hybrid Search & Reranking Engine

A production-ready Hybrid Search Engine integrating Dense Vector Search (Qdrant), Sparse Keyword Search (BM25), Reciprocal Rank Fusion (RRF), Cross-Encoder Neural Reranking, and an interactive, animated modern web UI.

---

## Key Features

- **Multi-Modal Retrieval**: Combines semantic deep learning (Dense Vectors via `all-MiniLM-L6-v2`) with exact term matching (Sparse Lexical BM25).
- **Contextual Neural Reranking**: Second-stage Cross-Encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) scores query-document pairs for superior precision.
- **Calibrated Percentage Match**: Automatically transforms unconstrained neural logits into calibrated relevance percentages ($\ge 85\%$ High, $60\%-84\%$ Medium, $<60\%$ Low) via Sigmoid transformation ($\sigma(x) = \frac{1}{1 + e^{-x}}$).
- **Interactive Modern UI with Background Animations**:
  - **Dynamic Particle Constellation**: Responsive canvas-based particle network interacting with mouse proximity.
  - **Ambient Glowing Nebulas & Cyber Grid**: Multi-hue floating gradient orbs and subtle dot-matrix background.
  - **Interactive Search Mode Switcher**: Toggle dynamically between **⚡ Hybrid (RRF + Reranker)**, **🧠 Neural Vector**, and **🔍 BM25 Keyword**.
  - **Real-Time Pipeline Telemetry**: Visual latency breakdown bar detailing execution times for BM25, Dense retrieval, RRF merge, and Cross-Encoder reranking.
  - **Drag-and-Drop Ingestion**: Upload PDFs, Markdown, TXT, CSV, or JSON documents with instant file preview and animated chunk indexing feedback.
  - **Result Card Interactions**: One-click "Copy Chunk" button, query word highlighting, and document type tagging.
- **Production Hardened**: In-memory Qdrant support, query embedding caching (`functools.lru_cache`), and Cross-Encoder graceful degradation (timeout fallback).
- **Benchmarking Suite**: Asynchronous load-testing script to monitor multi-stage latencies (Avg, Median, P95, P99).

---

## Tech Stack

- **API Framework**: FastAPI, Uvicorn, Pydantic, Pydantic-Settings
- **Vector Database**: Qdrant (`qdrant-client`, in-memory or networked)
- **Machine Learning & NLP**: `sentence-transformers`, `rank-bm25`, `pypdf`, `torch`
- **Frontend**: Vanilla HTML5, CSS3 Glassmorphism, Modern Vanilla JS with Canvas Particle Engine
- **Testing**: `pytest`, `httpx`

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

- **Web UI**: Visit `http://localhost:8000` (or `http://127.0.0.1:8000`)
- **API Documentation (Swagger UI)**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`

### 2. Run Automated Tests

```bash
pytest tests/
```

### 3. Run Benchmarks

To load-test the API and evaluate pipeline latencies (1,000 synthetic documents, 100 sequential queries, 50 concurrent queries):

```bash
python scripts/benchmark.py
```

Results are printed to the console as a Markdown table and exported to `results/benchmark_report.json`.

---

## 📡 API Endpoints

### `POST /search`
Executes search using the specified mode (`hybrid`, `vector`, or `keyword`) with optional metadata filtering and returns scored chunks with stage latencies.

**Request Body:**
```json
{
  "query": "IT Business Automation",
  "search_mode": "hybrid",
  "top_k": 5,
  "metadata_filters": {
    "category": "upload"
  }
}
```

**Response Body:**
```json
{
  "query": "IT Business Automation",
  "results": [
    {
      "chunk_id": "b174b148-...",
      "document_id": "job_description.pdf",
      "text": "IT Business Automation Intern...",
      "metadata": { "category": "upload" },
      "score": 7.5331
    }
  ],
  "latencies": {
    "bm25_search_ms": 0.12,
    "dense_search_ms": 8.65,
    "rrf_merge_ms": 0.04,
    "cross_encoder_ms": 248.15,
    "total_ms": 257.03
  }
}
```

### `POST /upload`
Uploads files (`.pdf`, `.txt`, `.md`, `.csv`, `.json`) using `multipart/form-data`. PDFs are parsed page-by-page via `pypdf`, chunked with overlap, and indexed simultaneously into Qdrant and BM25.

### `POST /ingest`
Direct JSON document ingestion endpoint.
```json
{
  "document_id": "doc-1",
  "text": "FastAPI is a modern, fast web framework...",
  "metadata": { "category": "tech" }
}
```

### `GET /health`
Returns system health, Qdrant connectivity, BM25 index status, and total indexed chunks.
