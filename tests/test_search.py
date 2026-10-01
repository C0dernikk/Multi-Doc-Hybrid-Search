import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.core.indexer import indexer

client = TestClient(app)

@pytest.fixture(autouse=True)
def clear_indices():
    # Setup - clear before each test
    indexer.chunks = []
    indexer.bm25 = None
    indexer.qdrant.recreate_collection(
        collection_name=indexer.collection_name,
        vectors_config=indexer.qdrant.get_collection(indexer.collection_name).config.params.vectors
    )
    yield
    # Teardown

def test_health_empty():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["total_chunks"] == 0
    assert response.json()["bm25_status"] == "empty"

def test_ingest_and_search():
    # Ingest document 1
    ingest_res = client.post("/ingest", json={
        "document_id": "doc_1",
        "text": "FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.",
        "metadata": {"category": "tech", "type": "docs"}
    })
    assert ingest_res.status_code == 200
    assert ingest_res.json()["chunks_processed"] > 0

    # Ingest document 2
    ingest_res2 = client.post("/ingest", json={
        "document_id": "doc_2",
        "text": "The quick brown fox jumps over the lazy dog. This is a common test sentence used to check fonts.",
        "metadata": {"category": "general"}
    })
    assert ingest_res2.status_code == 200

    # Search (Hybrid)
    search_res = client.post("/search", json={
        "query": "Python web framework",
        "search_mode": "hybrid",
        "top_k": 1
    })
    if search_res.status_code != 200:
        print("Error details:", search_res.json())
    assert search_res.status_code == 200
    data = search_res.json()
    assert len(data["results"]) == 1
    assert "FastAPI" in data["results"][0]["text"]
    assert data["results"][0]["document_id"] == "doc_1"

    # Search with exact keyword matching
    search_res_kw = client.post("/search", json={
        "query": "lazy dog",
        "search_mode": "keyword",
        "top_k": 1
    })
    assert search_res_kw.status_code == 200
    assert len(search_res_kw.json()["results"]) > 0
    assert "lazy dog" in search_res_kw.json()["results"][0]["text"]

    # Test Metadata Filter
    search_res_filter = client.post("/search", json={
        "query": "Python",
        "search_mode": "hybrid",
        "top_k": 5,
        "metadata_filters": {"category": "general"}
    })
    assert search_res_filter.status_code == 200
    data_filter = search_res_filter.json()
    assert len(data_filter["results"]) == 1
    assert data_filter["results"][0]["document_id"] == "doc_2"
