from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import io
import pypdf
from app.models.schemas import IngestRequest, IngestResponse, SearchRequest, SearchResponse, HealthResponse, SearchResultChunk
from app.core.chunker import chunk_text
from app.core.indexer import indexer
from app.core.search import execute_search_with_latencies
import time
from app.models.schemas import SearchLatencies
from app.services.reranker import reranker

app = FastAPI(title="Hybrid Search Engine")

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_dir), name="static")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(frontend_dir, "index.html"))

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Document text cannot be empty")
        
    chunks = chunk_text(
        text=request.text, 
        document_id=request.document_id, 
        metadata=request.metadata
    )
    
    try:
        indexer.add_chunks(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return IngestResponse(status="success", chunks_processed=len(chunks))

@app.post("/upload", response_model=IngestResponse)
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    
    if file.filename.lower().endswith(".pdf"):
        try:
            pdf_reader = pypdf.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {str(e)}")
    else:
        try:
            text = content.decode("utf-8")
        except Exception:
            raise HTTPException(status_code=400, detail="Only text-based files and PDFs are currently supported")
        
    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty")
        
    chunks = chunk_text(
        text=text, 
        document_id=file.filename, 
        metadata={"category": "upload"}
    )
    
    try:
        indexer.add_chunks(chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return IngestResponse(status="success", chunks_processed=len(chunks))

@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
        
    try:
        start_total = time.time()
        # 1. First stage retrieval (Sparse, Dense, or Hybrid)
        candidates, latencies_dict = execute_search_with_latencies(
            query=request.query, 
            search_mode=request.search_mode.value, 
            top_k=request.top_k if request.search_mode != "hybrid" else 20, 
            metadata_filters=request.metadata_filters
        )
        
        # 2. Second stage reranking
        cross_encoder_ms = 0.0
        if candidates and request.search_mode == "hybrid":
             t0 = time.time()
             reranked = reranker.rerank(request.query, candidates, top_k=request.top_k)
             cross_encoder_ms = (time.time() - t0) * 1000
        elif candidates:
             # Just sort by score if not hybrid, or slice top_k
             reranked = sorted(candidates, key=lambda x: x['score'], reverse=True)[:request.top_k]
        else:
             reranked = []
        
        # 3. Format response
        results = []
        for item in reranked:
            chunk = item['chunk']
            final_score = item.get('rerank_score', item['score'])
            results.append(SearchResultChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                text=chunk.text,
                metadata=chunk.metadata,
                score=final_score
            ))
            
        total_ms = (time.time() - start_total) * 1000
        
        latencies = SearchLatencies(
            bm25_search_ms=latencies_dict.get("bm25_search_ms", 0.0),
            dense_search_ms=latencies_dict.get("dense_search_ms", 0.0),
            rrf_merge_ms=latencies_dict.get("rrf_merge_ms", 0.0),
            cross_encoder_ms=cross_encoder_ms,
            total_ms=total_ms
        )
            
        return SearchResponse(query=request.query, results=results, latencies=latencies)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=HealthResponse)
async def health_check():
    qdrant_status = "ok"
    try:
        indexer.qdrant.get_collections()
    except Exception:
        qdrant_status = "error"
        
    return HealthResponse(
        status="ok",
        qdrant_status=qdrant_status,
        bm25_status="ok" if indexer.bm25 is not None else "empty",
        total_chunks=len(indexer.chunks)
    )
