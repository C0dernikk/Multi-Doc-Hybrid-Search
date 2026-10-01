# Developer Reference & Interview Guide

This document provides a comprehensive walkthrough of the project directory structure, code components, and a curated list of 50 interview questions and answers relevant to this specific hybrid search engine.

---

## Part 1: Project Directory & Code Walkthrough

### `/app` Directory
The core application folder containing the FastAPI application and business logic.
- **`__init__.py`**: Marks the directory as a Python package.
- **`main.py`**: The FastAPI entry point. Defines endpoints (`/ingest`, `/upload`, `/search`, `/health`), mounts the frontend, and acts as the orchestrator linking chunking, indexing, and reranking. It also calculates request latencies.
- **`config.py`**: Pydantic `BaseSettings` that manage environment variables (Qdrant host/port, model names, memory limits, and fusion parameters).

### `/app/core` Directory
Handles the heavy lifting of data processing and database interactions.
- **`chunker.py`**: Contains `chunk_text()`. It implements a fixed-size sliding window (with overlap) algorithm to break large documents into manageable `Chunk` objects.
- **`indexer.py`**: The `Indexer` singleton class. Manages the Qdrant vector database connection (with exponential backoff retries), computes dense embeddings using SentenceTransformers, and maintains the in-memory Sparse BM25 model. Implements `search_dense` (with LRU embedding caching) and `search_sparse`.
- **`search.py`**: Implements `reciprocal_rank_fusion (RRF)` to merge sparse and dense candidate lists. Exposes `execute_search_with_latencies` to handle the conditional logic of Keyword vs. Vector vs. Hybrid modes, while measuring step-by-step latency.

### `/app/models` Directory
- **`schemas.py`**: Pydantic models mapping the JSON API contracts (`SearchRequest`, `SearchResponse`, `IngestRequest`, `SearchLatencies`, etc.). Ensures data validation.

### `/app/services` Directory
- **`reranker.py`**: The `Reranker` singleton class. Loads the Cross-Encoder model (`cross-encoder/ms-marco-MiniLM-L-6-v2`) and performs secondary reranking of fused candidates. Implements a thread pool executor with a configurable timeout for graceful degradation under heavy load.

### `/frontend` Directory
- Contains static assets (`index.html`, etc.) for the web UI.

### `/scripts` Directory
- **`benchmark.py`**: An asynchronous load-testing script using `httpx`. It generates 1,000 synthetic chunks, pushes them to the ingest endpoint, and then executes 100 sequential and 50 concurrent search queries, outputting detailed P95/P99 latency metrics.

### `/results` Directory
- Automatically created by `benchmark.py` to store JSON-formatted load-test reports (`benchmark_report.json`).

### Root Files
- **`requirements.txt`**: Python dependencies (`fastapi`, `qdrant-client`, `sentence-transformers`, `rank-bm25`, etc.).
- **`README.md` & `ARCHITECTURE.md`**: Project documentation and architecture diagrams.

---

## Part 2: 50 Interview Questions & Answers

### System Architecture & Design
**1. What is a Hybrid Search Engine?**
It's a retrieval system combining semantic search (understanding meaning via dense vectors) and keyword search (exact term matching via sparse vectors/BM25) to provide the most relevant results.

**2. Why use FastAPI for this project?**
FastAPI is built on Starlette and Pydantic, providing high performance, native async support (crucial for I/O bound DB calls), and automatic OpenAPI documentation.

**3. What is the role of the Cross-Encoder in the architecture?**
It acts as a second-stage reranker. While the first stage (Dense+Sparse) retrieves broadly relevant candidates quickly, the Cross-Encoder scores the exact query-document pairs contextually for highly precise sorting.

**4. Why decouple Qdrant into its own container?**
Separating state (Qdrant DB) from stateless compute (FastAPI API) allows the API to scale horizontally, ensures data persistence via volumes, and aligns with microservice best practices.

**5. How does the system achieve Graceful Degradation?**
The Reranker uses a `ThreadPoolExecutor` with a timeout. If the cross-encoder takes too long (e.g., under heavy concurrent load), it catches the `TimeoutError` and returns the un-reranked (but still highly relevant) RRF candidates instead of failing the request.

**6. Why does the system cache embeddings?**
The `Indexer._encode_query` method is decorated with `@lru_cache`, which returns the cached vector instantly for repeated queries, saving CPU/GPU cycles.

**7. How do you monitor performance bottlenecks in this architecture?**
The API measures latencies at each stage (BM25, Dense, RRF, Cross-Encoder) using `time.time()` and returns them in the `SearchLatencies` schema.

**8. Why is the `chunker.py` necessary?**
Transformer models have token limits (e.g., 512 tokens). Chunking breaks long documents into smaller, overlapping segments so they can be properly embedded and retrieved without truncation.

### Embeddings & Vector Search (Dense)
**11. What is a Dense Vector?**
A continuous array of floats representing the semantic meaning of text in a high-dimensional space.

**12. Which model is used for dense embeddings and why?**
`all-MiniLM-L6-v2`. It is lightweight, fast, and provides an excellent balance between embedding quality and inference speed for real-time APIs.

**13. What distance metric does Qdrant use in this project?**
Cosine Similarity (`Distance.COSINE`), which measures the cosine of the angle between two vectors, making it excellent for comparing semantic text embeddings regardless of text length.

**14. What happens when a dense query is exactly the same as a previous one?**
The `Indexer._encode_query` method is decorated with `@lru_cache`, which returns the cached vector instantly, saving CPU/GPU cycles.

**15. How are metadata filters handled in the dense search?**
They are converted into Qdrant `FieldCondition` objects (MatchValue) and passed as a `Filter` to the `query_points` method.

**16. Why do we need payload in Qdrant?**
Payload stores the original chunk text and metadata, allowing the search result to be immediately usable without a secondary database lookup.

**17. What is the embedding dimensionality of `all-MiniLM-L6-v2`?**
384 dimensions.

**18. What is Qdrant?**
A high-performance open-source vector similarity search engine written in Rust.

**19. Can this system run entirely in memory?**
Yes, `config.py` allows setting `qdrant_memory_only = True` for testing without a dedicated Qdrant container.

**20. What is the downside of dense search alone?**
It can miss exact keyword matches (e.g., specific serial numbers, unique names) because it focuses on general semantic meaning.

### Keyword Search (Sparse / BM25)
**21. What is BM25?**
Best Matching 25 is a ranking function used by search engines to estimate the relevance of documents to a given search query based on term frequencies and inverse document frequencies (TF-IDF).

**22. Why use `BM25Plus` specifically?**
It slightly modifies the BM25 formula to solve the issue of overly penalizing long documents, establishing a lower bound on term frequency normalization.

**23. How is the sparse index updated in this project?**
Currently, chunks are stored in an in-memory list, and the `BM25Plus` corpus is rebuilt on ingestion. 

**24. What is a limitation of the current BM25 implementation?**
It resides entirely in Python memory. In a large-scale distributed production setup, this would need to be moved to a persistent search engine like Elasticsearch, or Qdrant's native sparse vectors.

**25. How does the system tokenize text for BM25?**
It uses a simple Regex `r'\w+'` and converts text to lowercase.

**26. How are metadata filters applied to the sparse search?**
After BM25 scoring, the system iterates over the top candidates and manually filters out chunks whose metadata dict does not match the request filters.

**27. Why do we filter *after* BM25 scoring rather than before?**
Because the `rank_bm25` library expects a static corpus. Post-filtering is easier to implement in-memory, though less efficient at massive scale.

**28. If a word in the query isn't in the BM25 corpus, what happens?**
BM25 ignores the unknown token and scores based on the known tokens.

**29. Why keep sparse search if dense search is "smarter"?**
Sparse search dominates in zero-shot exact keyword matching (acronyms, IDs), providing a critical safety net for dense embeddings.

**30. What data structure holds the chunks for BM25?**
A standard Python `List[Chunk]`.

### Fusion & Re-Ranking
**31. What is Reciprocal Rank Fusion (RRF)?**
A method for combining multiple ranked result lists without requiring normalized scores. It scores items based on their rank position: `Score = 1 / (k + rank)`.

**32. Why is RRF preferred over score normalization?**
BM25 scores and Cosine distances have completely different scales and distributions. RRF mathematically side-steps this by only looking at the rank order.

**33. What is the constant `k` in RRF?**
It mitigates the impact of high ranks by flattening the curve. In this project, `k=60` is used, a common industry standard.

**34. What is a Bi-Encoder vs. a Cross-Encoder?**
Bi-Encoders (SentenceTransformers) encode the query and document separately into vectors. Cross-Encoders pass the query and document together into the transformer simultaneously for deep attention-based scoring.

**35. Why not use the Cross-Encoder for the entire database?**
Cross-Encoders are incredibly computationally expensive. Scoring 1 million documents against a query would take minutes/hours. Bi-encoders + Vector DBs do it in milliseconds.

**36. Which model is used for the Cross-Encoder?**
`cross-encoder/ms-marco-MiniLM-L-6-v2`

**37. How many candidates are passed to the Cross-Encoder?**
The top 20 candidates resulting from the RRF merge are passed to the Cross-Encoder.

**38. How is the final `top_k` enforced?**
After the Cross-Encoder scores the 20 candidates, they are sorted by `rerank_score`, and the top `N` (based on user request) are sliced and returned.

**39. What happens if the Cross-Encoder fails entirely?**
The `try/except` block in `Reranker.rerank` catches the exception, logs it, and returns the fallback RRF candidates.

**40. Why prepare pairs as `[query, document_text]`?**
Because the HuggingFace Cross-Encoder expects inputs as pairs of sequences to compute cross-attention between them.

### DevOps, Hardening & Concurrency
**41. How does the API achieve concurrency?**
FastAPI uses `async/await`. I/O bound operations (like HTTP requests in the benchmark) yield control back to the event loop.

**42. Why use `ThreadPoolExecutor` for the Cross-Encoder?**
The HuggingFace `predict()` method is synchronous and CPU-bound. Running it directly would block the FastAPI event loop. A thread pool prevents event loop starvation.

**43. Why is `max_workers=1` in the `ThreadPoolExecutor`?**
To prevent CPU contention and memory spikes on a single node, ensuring requests are queued rather than competing for parallel heavy ML inference.

**44. What does `WEB_CONCURRENCY` do when running Uvicorn?**
It dictates the number of Uvicorn worker processes to spawn, enabling true multi-core processing for the API.

**45. What is the purpose of the `/health` endpoint?**
It is used by load balancers and system monitors to verify the API is alive and the Qdrant connection is established before routing traffic to it.

**46. How does the benchmark script test concurrency?**
It uses `asyncio.gather(*tasks)` to fire 50 HTTP POST requests to the `/search` endpoint simultaneously.

**47. What does P95 and P99 latency mean?**
P95 (95th percentile) means 95% of requests completed faster than this time. P99 is the 99th percentile. They are critical for understanding the "long tail" of slow requests.

**48. Why does the benchmark script have a "warmup" query?**
To initialize any lazy-loaded models, populate the LRU cache, and establish HTTP connection pools so they don't skew the actual benchmark timings.

**49. How is the benchmark report exported?**
It is dumped into a `results/benchmark_report.json` file using Python's `json.dump`.

**50. What is a synthetic chunk in the benchmark?**
It's randomly generated text constructed from a list of keywords to simulate data ingestion without needing an external dataset dependency.
