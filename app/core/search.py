import time
from typing import List, Dict, Any, Optional, Tuple
from app.core.indexer import indexer
from app.config import settings

def reciprocal_rank_fusion(dense_results: List[Dict[str, Any]], sparse_results: List[Dict[str, Any]], k: int = 60) -> List[Dict[str, Any]]:
    # dense_results and sparse_results are lists of dicts with 'chunk' (Chunk object) and 'score'
    
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Any] = {}
    
    # Process dense
    for rank, item in enumerate(dense_results):
        chunk_id = item['chunk'].chunk_id
        chunk_map[chunk_id] = item['chunk']
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        
    # Process sparse
    for rank, item in enumerate(sparse_results):
        chunk_id = item['chunk'].chunk_id
        chunk_map[chunk_id] = item['chunk']
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
        
    # Sort by RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [{"chunk": chunk_map[chunk_id], "score": score} for chunk_id, score in sorted_items]

def execute_search_with_latencies(query: str, search_mode: str, top_k: int, metadata_filters: Optional[Dict[str, Any]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    latencies = {"bm25_search_ms": 0.0, "dense_search_ms": 0.0, "rrf_merge_ms": 0.0}
    
    if search_mode == "keyword":
        t0 = time.time()
        results = indexer.search_sparse(query, top_k=top_k, metadata_filters=metadata_filters)
        latencies["bm25_search_ms"] = (time.time() - t0) * 1000
        return results, latencies
    elif search_mode == "vector":
        t0 = time.time()
        results = indexer.search_dense(query, top_k=top_k, metadata_filters=metadata_filters)
        latencies["dense_search_ms"] = (time.time() - t0) * 1000
        return results, latencies
    elif search_mode == "hybrid":
        t0 = time.time()
        dense_candidates = indexer.search_dense(query, top_k=20, metadata_filters=metadata_filters)
        latencies["dense_search_ms"] = (time.time() - t0) * 1000
        
        t0 = time.time()
        sparse_candidates = indexer.search_sparse(query, top_k=20, metadata_filters=metadata_filters)
        latencies["bm25_search_ms"] = (time.time() - t0) * 1000
        
        t0 = time.time()
        merged = reciprocal_rank_fusion(dense_candidates, sparse_candidates, k=settings.rrf_k)
        latencies["rrf_merge_ms"] = (time.time() - t0) * 1000
        return merged[:20], latencies
    
    return [], latencies
