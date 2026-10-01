from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Plus
from typing import List, Dict, Any, Optional
from app.core.chunker import Chunk
from app.config import settings
import uuid
import time
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class Indexer:
    def __init__(self):
        # Qdrant client
        if settings.qdrant_memory_only:
            self.qdrant = QdrantClient(":memory:")
        else:
            self.qdrant = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        
        self.collection_name = "hybrid_chunks"
        self.embedding_model = SentenceTransformer(settings.dense_model_name)
        
        # Ensure collection exists
        try:
            self.qdrant.get_collection(self.collection_name)
        except Exception:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_model.get_embedding_dimension(), 
                    distance=Distance.COSINE
                ),
            )

        # BM25 Sparse Index state
        self.chunks: List[Chunk] = []
        self.bm25: Optional[BM25Plus] = None

    def tokenize(self, text: str) -> List[str]:
        import re
        return re.findall(r'\w+', text.lower())

    def add_chunks(self, chunks: List[Chunk]):
        if not chunks:
            return
            
        # Dense Indexing
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)
        
        points = []
        for chunk, emb in zip(chunks, embeddings):
            payload = chunk.model_dump()
            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=emb.tolist(),
                    payload=payload
                )
            )
            
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        # Sparse Indexing
        self.chunks.extend(chunks)
        tokenized_corpus = [self.tokenize(chunk.text) for chunk in self.chunks]
        self.bm25 = BM25Plus(tokenized_corpus)
        
    @lru_cache(maxsize=1024)
    def _encode_query(self, query: str) -> List[float]:
        return self.embedding_model.encode(query).tolist()

    def search_dense(self, query: str, top_k: int = 20, metadata_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        query_vector = self._encode_query(query)
        
        qdrant_filter = None
        if metadata_filters:
            must_conditions = []
            for k, v in metadata_filters.items():
                must_conditions.append(
                    FieldCondition(key=f"metadata.{k}", match=MatchValue(value=v))
                )
            qdrant_filter = Filter(must=must_conditions)

        query_response = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k
        )
        results = query_response.points
        
        return [{"chunk": Chunk(**hit.payload), "score": hit.score} for hit in results]

    def search_sparse(self, query: str, top_k: int = 20, metadata_filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        scored_chunks = list(zip(self.chunks, scores))
        # Apply metadata filters
        if metadata_filters:
            filtered_scored_chunks = []
            for chunk, score in scored_chunks:
                match = True
                for k, v in metadata_filters.items():
                    if chunk.metadata.get(k) != v:
                        match = False
                        break
                if match:
                    filtered_scored_chunks.append((chunk, score))
            scored_chunks = filtered_scored_chunks
            
        # Sort and get top_k
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        top_scored = scored_chunks[:top_k]
        
        return [{"chunk": item[0], "score": float(item[1])} for item in top_scored if item[1] > 0.0]

# Singleton instance
indexer = Indexer()
