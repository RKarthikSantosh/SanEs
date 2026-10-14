"""Compare BM25-only, dense-only, hybrid (RRF) and hybrid + Cohere re-ranking on a BEIR dataset.

Usage (from backend/):  .venv/bin/python -m experiments.run_retrieval scifact [--max-queries 200] [--no-rerank]
"""
import argparse
import re
import statistics
import time

import numpy as np
import pytrec_eval
from scipy import stats

from experiments.common import DEPTH, CohereReranker, Retrievers, load_beir, rrf, save_json

CONFIGS = ["bm25", "dense", "hybrid", "hybrid_rerank"]
EXACT_TERM = re.compile(r"\b\w*\d\w*\b|\b[A-Z]{2,}\b|\b[A-Z][a-z]+[A-Z]\w*\b")


def mrr_at(ranking: list[str], relevant: dict[str, int], k: int = 10) -> float:
    for rank, doc_id in enumerate(ranking[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def success_at(ranking: list[str], relevant: dict[str, int], k: int) -> float:
    return float(any(doc_id in relevant for doc_id in ranking[:k]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    corpus, queries, qrels = load_beir(args.dataset, max_queries=args.max_queries)
    print(f"{args.dataset}: {len(corpus)} docs, {len(queries)} queries", flush=True)
    retrievers = Retrievers(args.dataset, corpus)
    reranker = None if args.no_rerank else CohereReranker(args.dataset)

    runs = {config: {} for config in CONFIGS if config != "hybrid_rerank" or reranker}
    latency = {config: [] for config in runs}
    rankings = {config: {} for config in runs}

    for count, (query_id, query) in enumerate(queries.items(), start=1):
        start = time.perf_counter()
        bm25 = retrievers.bm25_search(query)
        bm25_time = time.perf_counter() - start

        start = time.perf_counter()
        dense = retrievers.dense_search(query)
        dense_time = time.perf_counter() - start

        start = time.perf_counter()
        hybrid = rrf([dense, bm25])
        fusion_time = time.perf_counter() - start

        results = {"bm25": bm25, "dense": dense, "hybrid": hybrid}
        latency["bm25"].append(bm25_time)
        latency["dense"].append(dense_time)
        latency["hybrid"].append(bm25_time + dense_time + fusion_time)

        if reranker:
            reranked, rerank_time = reranker.rerank(query_id, query, hybrid, corpus)
            results["hybrid_rerank"] = reranked
            latency["hybrid_rerank"].append(bm25_time + dense_time + fusion_time + rerank_time)

        for config, ranked in results.items():
            # strictly decreasing scores so pytrec_eval keeps our rank order
            runs[config][query_id] = {doc_id: float(DEPTH - position) for position, (doc_id, _) in enumerate(ranked)}
            rankings[config][query_id] = [doc_id for doc_id, _ in ranked]

        if count % 50 == 0:
            print(f"  {count}/{len(queries)} queries", flush=True)

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.5,10,20"})
    per_query = {config: evaluator.evaluate(run) for config, run in runs.items()}

    summary, per_query_ndcg = {}, {}
    for config in runs:
        scores = per_query[config]
        query_ids = sorted(scores)
        per_query_ndcg[config] = np.array([scores[q]["ndcg_cut_10"] for q in query_ids])
        summary[config] = {
            "nDCG@10": float(np.mean(per_query_ndcg[config])),
            "Recall@5": float(np.mean([scores[q]["recall_5"] for q in query_ids])),
            "Recall@10": float(np.mean([scores[q]["recall_10"] for q in query_ids])),
            "Recall@20": float(np.mean([scores[q]["recall_20"] for q in query_ids])),
            "MRR@10": float(np.mean([mrr_at(rankings[config][q], qrels[q]) for q in query_ids])),
            "Success@5": float(np.mean([success_at(rankings[config][q], qrels[q], 5) for q in query_ids])),
            "latency_ms_mean": 1000 * statistics.mean(latency[config]),
            "latency_ms_p95": 1000 * float(np.percentile(latency[config], 95)),
        }

    # paired significance tests on per-query nDCG@10
    significance = {}
    for system in [config for config in ("hybrid", "hybrid_rerank") if config in runs]:
        for baseline in ("bm25", "dense"):
            a, b = per_query_ndcg[system], per_query_ndcg[baseline]
            diff = a - b
            significance[f"{system}_vs_{baseline}"] = {
                "mean_diff": float(diff.mean()),
                "t_test_p": float(stats.ttest_rel(a, b).pvalue),
                "wilcoxon_p": float(stats.wilcoxon(a, b).pvalue) if np.any(diff != 0) else 1.0,
                "wins": int((diff > 1e-9).sum()),
                "ties": int((np.abs(diff) <= 1e-9).sum()),
                "losses": int((diff < -1e-9).sum()),
            }

    # how differently do BM25 and dense search behave?
    overlap, only_bm25, only_dense, neither = [], 0, 0, 0
    for query_id in queries:
        bm25_top, dense_top = rankings["bm25"][query_id], rankings["dense"][query_id]
        overlap.append(len(set(bm25_top[:10]) & set(dense_top[:10])) / 10)
        bm25_hit = any(doc_id in qrels[query_id] for doc_id in bm25_top)
        dense_hit = any(doc_id in qrels[query_id] for doc_id in dense_top)
        only_bm25 += bm25_hit and not dense_hit
        only_dense += dense_hit and not bm25_hit
        neither += not bm25_hit and not dense_hit
    complementarity = {
        "mean_top10_overlap": float(np.mean(overlap)),
        "queries_only_bm25_finds_relevant_in_top20": only_bm25,
        "queries_only_dense_finds_relevant_in_top20": only_dense,
        "queries_neither_finds_relevant_in_top20": neither,
    }

    # exact-term queries (numbers, acronyms, codes) versus ordinary natural-language queries
    by_type = {}
    query_ids = sorted(per_query["bm25"])
    exact = [q for q in query_ids if EXACT_TERM.search(queries[q])]
    other = [q for q in query_ids if not EXACT_TERM.search(queries[q])]
    for label, group in (("exact_term", exact), ("natural_language", other)):
        if group:
            by_type[label] = {"count": len(group)} | {
                config: float(np.mean([per_query[config][q]["ndcg_cut_10"] for q in group])) for config in runs
            }

    # examples where one retriever fails and the hybrid still succeeds (used for the qualitative discussion)
    examples = []
    for query_id in query_ids:
        bm25_ok = success_at(rankings["bm25"][query_id], qrels[query_id], 5)
        dense_ok = success_at(rankings["dense"][query_id], qrels[query_id], 5)
        best = "hybrid_rerank" if "hybrid_rerank" in runs else "hybrid"
        if bm25_ok != dense_ok and success_at(rankings[best][query_id], qrels[query_id], 5) and len(examples) < 6:
            examples.append({"query": queries[query_id], "bm25_hit@5": bm25_ok, "dense_hit@5": dense_ok})

    payload = {
        "dataset": args.dataset,
        "documents": len(corpus),
        "queries": len(queries),
        "depth": DEPTH,
        "index_build_seconds": {"bm25": retrievers.bm25_build_seconds, "embeddings": retrievers.embed_seconds},
        "summary": summary,
        "significance": significance,
        "complementarity": complementarity,
        "by_query_type": by_type,
        "examples": examples,
        "cohere_calls_this_run": reranker.calls if reranker else 0,
    }
    path = save_json(f"retrieval_{args.dataset}.json", payload)

    print(f"\n{'config':15s} " + " ".join(f"{metric:>10s}" for metric in next(iter(summary.values()))))
    for config, metrics in summary.items():
        print(f"{config:15s} " + " ".join(f"{value:10.4f}" for value in metrics.values()))
    print("significance:", significance)
    print("complementarity:", complementarity)
    print("by type:", by_type)
    print("saved", path)


if __name__ == "__main__":
    main()
