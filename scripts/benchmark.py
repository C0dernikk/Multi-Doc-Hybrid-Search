import asyncio
import httpx
import json
import time
import os
import statistics
import random

API_URL = "http://localhost:8000"

def generate_synthetic_data(num_chunks: int = 1000):
    chunks = []
    for i in range(num_chunks):
        words = ["search", "hybrid", "vector", "sparse", "dense", "AI", "machine", "learning", "data", "benchmark"]
        text = " ".join(random.choices(words, k=20))
        chunks.append({
            "document_id": f"doc_{i}",
            "text": text,
            "metadata": {"source": "synthetic"}
        })
    return chunks

async def ingest_data(chunks, client):
    print(f"Ingesting {len(chunks)} synthetic chunks...")
    for chunk in chunks:
        resp = await client.post(f"{API_URL}/ingest", json=chunk)
        resp.raise_for_status()
    print("Ingestion complete.")

async def run_query(client, query: str):
    payload = {
        "query": query,
        "search_mode": "hybrid",
        "top_k": 5
    }
    resp = await client.post(f"{API_URL}/search", json=payload)
    resp.raise_for_status()
    return resp.json().get("latencies", {})

async def benchmark_sequential(client, queries):
    print(f"Running {len(queries)} sequential queries...")
    latencies_list = []
    for q in queries:
        latencies = await run_query(client, q)
        latencies_list.append(latencies)
    return latencies_list

async def benchmark_concurrent(client, queries):
    print(f"Running {len(queries)} concurrent queries...")
    tasks = [run_query(client, q) for q in queries]
    results = await asyncio.gather(*tasks)
    return results

def compute_stats(latencies_list, name):
    metrics = ["bm25_search_ms", "dense_search_ms", "rrf_merge_ms", "cross_encoder_ms", "total_ms"]
    stats = {"test": name}
    
    if not latencies_list:
        return stats

    for m in metrics:
        values = [l.get(m, 0.0) for l in latencies_list]
        values.sort()
        
        avg_val = statistics.mean(values)
        med_val = statistics.median(values)
        
        # P95 and P99 calculation
        idx95 = int(len(values) * 0.95)
        idx99 = int(len(values) * 0.99)
        
        p95_val = values[idx95] if idx95 < len(values) else values[-1]
        p99_val = values[idx99] if idx99 < len(values) else values[-1]
        
        stats[m] = {
            "avg": avg_val,
            "median": med_val,
            "p95": p95_val,
            "p99": p99_val
        }
    return stats

def print_markdown_table(stats_seq, stats_conc):
    print("\n### Benchmark Results\n")
    print("| Metric | Seq Avg (ms) | Seq P95 (ms) | Conc Avg (ms) | Conc P95 (ms) |")
    print("|--------|--------------|--------------|---------------|---------------|")
    
    metrics = [
        ("BM25 Search", "bm25_search_ms"),
        ("Dense Vector Search", "dense_search_ms"),
        ("RRF Merge", "rrf_merge_ms"),
        ("Cross-Encoder", "cross_encoder_ms"),
        ("Total End-to-End", "total_ms")
    ]
    
    for label, key in metrics:
        seq_avg = stats_seq.get(key, {}).get("avg", 0.0)
        seq_p95 = stats_seq.get(key, {}).get("p95", 0.0)
        conc_avg = stats_conc.get(key, {}).get("avg", 0.0)
        conc_p95 = stats_conc.get(key, {}).get("p95", 0.0)
        
        print(f"| {label} | {seq_avg:.2f} | {seq_p95:.2f} | {conc_avg:.2f} | {conc_p95:.2f} |")

async def main():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Check health
        try:
            r = await client.get(f"{API_URL}/health")
            r.raise_for_status()
        except Exception as e:
            print(f"Error connecting to API: {e}")
            return
            
        chunks = generate_synthetic_data(1000)
        await ingest_data(chunks, client)
        
        queries = [" ".join(random.choices(["search", "hybrid", "vector", "data"], k=3)) for _ in range(100)]
        
        # Warmup
        await run_query(client, "test warmup")
        
        # 100 Sequential
        seq_latencies = await benchmark_sequential(client, queries[:100])
        seq_stats = compute_stats(seq_latencies, "Sequential 100")
        
        # 50 Concurrent
        conc_latencies = await benchmark_concurrent(client, queries[:50])
        conc_stats = compute_stats(conc_latencies, "Concurrent 50")
        
        # Print table
        print_markdown_table(seq_stats, conc_stats)
        
        # Export to JSON
        os.makedirs("../results", exist_ok=True)
        report_path = os.path.join(os.path.dirname(__file__), "..", "results", "benchmark_report.json")
        with open(report_path, "w") as f:
            json.dump({
                "sequential": seq_stats,
                "concurrent": conc_stats
            }, f, indent=2)
        print(f"\nResults exported to {report_path}")

if __name__ == "__main__":
    asyncio.run(main())
