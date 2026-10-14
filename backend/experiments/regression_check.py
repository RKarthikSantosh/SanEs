"""CI retrieval regression gate on BEIR SciFact (no API keys needed).

Fails (exit code 1) when hybrid retrieval drops below a fixed nDCG@10 floor, or stops
beating both single retrievers. Usage: .venv/bin/python -m experiments.regression_check
"""
import sys

import numpy as np
import pytrec_eval

from experiments.common import DEPTH, Retrievers, load_beir, rrf

NDCG_FLOOR = 0.67  # measured 0.688 on 2026-10-14; allow a small margin for library changes


def main() -> int:
    corpus, queries, qrels = load_beir("scifact")
    retrievers = Retrievers("scifact", corpus)
    runs = {"bm25": {}, "dense": {}, "hybrid": {}}
    for query_id, query in queries.items():
        bm25, dense = retrievers.bm25_search(query), retrievers.dense_search(query)
        for name, ranked in (("bm25", bm25), ("dense", dense), ("hybrid", rrf([dense, bm25]))):
            runs[name][query_id] = {doc_id: float(DEPTH - position) for position, (doc_id, _) in enumerate(ranked)}

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10"})
    ndcg = {name: float(np.mean([s["ndcg_cut_10"] for s in evaluator.evaluate(run).values()])) for name, run in runs.items()}
    print({name: round(value, 4) for name, value in ndcg.items()})

    failures = []
    if ndcg["hybrid"] < NDCG_FLOOR:
        failures.append(f"hybrid nDCG@10 {ndcg['hybrid']:.4f} is below the floor {NDCG_FLOOR}")
    if ndcg["hybrid"] < max(ndcg["bm25"], ndcg["dense"]):
        failures.append("hybrid retrieval no longer beats both single retrievers")
    for failure in failures:
        print("REGRESSION:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
