import uuid
from typing import List, Dict, Any
from pydantic import BaseModel

class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    metadata: Dict[str, Any]

def chunk_text(text: str, document_id: str, metadata: Dict[str, Any], chunk_size: int = 500, overlap: int = 50) -> List[Chunk]:
    chunks = []
    start = 0
    # Handle empty text
    if not text:
        return chunks
        
    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]
        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            text=chunk_text,
            metadata=metadata.copy()
        )
        chunks.append(chunk)
        # Advance by chunk_size - overlap, but ensure we always advance
        advance = chunk_size - overlap
        if advance <= 0:
            advance = chunk_size
        start += advance
    return chunks
