from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from enum import Enum

class SearchLatencies(BaseModel):
    bm25_search_ms: float = 0.0
    dense_search_ms: float = 0.0
    rrf_merge_ms: float = 0.0
    cross_encoder_ms: float = 0.0
    total_ms: float = 0.0

class SearchMode(str, Enum):
    keyword = "keyword"
    vector = "vector"
    hybrid = "hybrid"

class IngestRequest(BaseModel):
    document_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestResponse(BaseModel):
    status: str
    chunks_processed: int

class SearchRequest(BaseModel):
    query: str
    search_mode: SearchMode = SearchMode.hybrid
    top_k: int = 5
    metadata_filters: Optional[Dict[str, Any]] = None

class SearchResultChunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any]
    score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultChunk]
    latencies: Optional[SearchLatencies] = None
    
class HealthResponse(BaseModel):
    status: str
    qdrant_status: str
    bm25_status: str
    total_chunks: int
