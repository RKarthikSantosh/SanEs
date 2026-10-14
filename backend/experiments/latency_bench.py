"""Clean latency measurement (run when nothing else is using the CPU).

Times BM25, dense search and RRF fusion per query on each BEIR dataset, and takes the Cohere
re-ranking time from the latencies recorded in the re-rank cache. No API calls are made.
Usage (from backend/): .venv/bin/python -m experiments.latency_bench
"""
import json
import statistics
import time

import numpy as np

from experiments.common import CACHE_DIR, Retrievers, load_beir, rrf, save_json

DATASETS = {"scifact": None, "nfcorpus": None, "fiqa": 200}


def main() -> None:
    report = {}
    for name, max_queries in DATASETS.items():
        corpus, queries, _ = load_beir(name, max_queries=max_queries)
        retrievers = Retrievers(name, corpus)
        for query in list(queries.values())[:10]:  # warm-up
            retrievers.bm25_search(query)
            retrievers.dense_search(query)

        timings = {"bm25": [], "dense": [], "fusion": []}
        for query in queries.values():
            start = time.perf_counter()
            bm25 = retrievers.bm25_search(query)
            timings["bm25"].append(time.perf_counter() - start)
            start = time.perf_counter()
            dense = retrievers.dense_search(query)
            timings["dense"].append(time.perf_counter() - start)
            start = time.perf_counter()
            rrf([dense, bm25])
            timings["fusion"].append(time.perf_counter() - start)

        cache = json.loads((CACHE_DIR / f"rerank_{name}.json").read_text())
        rerank = [entry["latency"] for key, entry in cache.items() if "|" in key and key.split("|")[0] in queries]

        ms = {key: 1000 * statistics.mean(values) for key, values in timings.items()}
        report[name] = {
            "documents": len(corpus),
            "queries": len(queries),
            "bm25_ms": ms["bm25"],
            "dense_ms": ms["dense"],
            "fusion_ms": ms["fusion"],
            "hybrid_sequential_ms": ms["bm25"] + ms["dense"] + ms["fusion"],
            "hybrid_parallel_ms": max(ms["bm25"], ms["dense"]) + ms["fusion"],
            "rerank_ms": 1000 * statistics.mean(rerank) if rerank else None,
            "rerank_ms_median": 1000 * float(np.median(rerank)) if rerank else None,
            "bm25_index_build_s": retrievers.bm25_build_seconds,
        }
        print(name, {key: round(value, 2) if isinstance(value, float) else value for key, value in report[name].items()}, flush=True)
    print("saved", save_json("latency.json", report))


if __name__ == "__main__":
    main()
