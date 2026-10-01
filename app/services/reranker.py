from sentence_transformers import CrossEncoder
from app.config import settings
from typing import List, Dict, Any
import concurrent.futures
import logging

logger = logging.getLogger(__name__)

class Reranker:
    def __init__(self):
        self.model = CrossEncoder(settings.cross_encoder_model_name)
        
    def _predict(self, pairs):
        return self.model.predict(pairs)

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5, timeout: float = 2.0) -> List[Dict[str, Any]]:
        if not candidates:
            return []
            
        # Prepare pairs
        pairs = [[query, item['chunk'].text] for item in candidates]
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._predict, pairs)
                scores = future.result(timeout=timeout)
                
            # Assign scores and sort
            for i, item in enumerate(candidates):
                item['rerank_score'] = float(scores[i])
                
            candidates.sort(key=lambda x: x['rerank_score'], reverse=True)
            return candidates[:top_k]
        except concurrent.futures.TimeoutError:
            logger.warning("Cross-Encoder reranking timed out. Falling back to raw candidates.")
            return candidates[:top_k]
        except Exception as e:
            logger.error(f"Cross-Encoder failed: {e}. Falling back to raw candidates.")
            return candidates[:top_k]

reranker = Reranker()
