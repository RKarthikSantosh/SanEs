"""Shared helpers for the retrieval and end-to-end experiments."""
import hashlib
import json
import os
import random
import re
import string
import time
from collections import Counter
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.retrieval.bm25_store import BM25Index

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATASETS_DIR = Path(os.getenv("DATASETS_DIR", BACKEND_DIR.parents[1] / "datasets"))
CACHE_DIR = BACKEND_DIR / "experiments" / "cache"
RESULTS_DIR = BACKEND_DIR / "experiments" / "results"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
DEPTH = 20  # every configuration returns a ranked list of 20 documents, as in the deployed system


# ---------------------------------------------------------------- datasets

def load_beir(name: str, split: str = "test", max_queries: int | None = None):
    root = DATASETS_DIR / name
    corpus = {}
    with open(root / "corpus.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            corpus[row["_id"]] = (row.get("title", "") + " " + row.get("text", "")).strip()

    qrels: dict[str, dict[str, int]] = {}
    with open(root / "qrels" / f"{split}.tsv", encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            query_id, doc_id, score = line.rstrip("\n").split("\t")
            if int(score) > 0:
                qrels.setdefault(query_id, {})[doc_id] = int(score)

    queries = {}
    with open(root / "queries.jsonl", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["_id"] in qrels:
                queries[row["_id"]] = row["text"]

    if max_queries and len(queries) > max_queries:
        chosen = sorted(random.Random(SEED).sample(sorted(queries), max_queries))
        queries = {query_id: queries[query_id] for query_id in chosen}
        qrels = {query_id: qrels[query_id] for query_id in chosen}
    return corpus, queries, qrels


def load_squad(sample_size: int):
    with open(DATASETS_DIR / "squad" / "dev-v1.1.json", encoding="utf-8") as handle:
        data = json.load(handle)["data"]

    corpus, questions = {}, []
    for article in data:
        for index, paragraph in enumerate(article["paragraphs"]):
            doc_id = f"{article['title']}#{index}"
            corpus[doc_id] = paragraph["context"]
            for qa in paragraph["qas"]:
                questions.append(
                    {
                        "id": qa["id"],
                        "question": qa["question"],
                        "answers": sorted({answer["text"] for answer in qa["answers"]}),
                        "gold_doc": doc_id,
                        "title": article["title"],
                    }
                )
    sample = random.Random(SEED).sample(questions, sample_size)
    return corpus, sample


# ---------------------------------------------------------------- retrievers

class Retrievers:
    """BM25, dense and hybrid retrieval over one corpus, using the same code paths as the backend."""

    def __init__(self, name: str, corpus: dict[str, str]):
        from app.embeddings.text_embeddings import get_embedding_model

        self.doc_ids = list(corpus)
        self.texts = [corpus[doc_id] for doc_id in self.doc_ids]
        self.settings = get_settings()

        start = time.perf_counter()
        self.bm25 = BM25Index(self.texts)
        self.bm25_build_seconds = time.perf_counter() - start

        self.model = get_embedding_model()
        cache = CACHE_DIR / f"emb_{name}_{len(self.doc_ids)}.npy"
        start = time.perf_counter()
        if cache.exists():
            self.embeddings = np.load(cache)
        else:
            self.embeddings = self.model.encode(
                self.texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True
            ).astype(np.float32)
            np.save(cache, self.embeddings)
        self.embed_seconds = time.perf_counter() - start

    def bm25_search(self, query: str, k: int = DEPTH) -> list[tuple[str, float]]:
        return [(self.doc_ids[index], score) for index, score in self.bm25.search(query, k)]

    def dense_search(self, query: str, k: int = DEPTH) -> list[tuple[str, float]]:
        # Exact cosine search over the same normalized vectors the backend stores in ChromaDB.
        # ChromaDB's HNSW index is approximate and its graph differs slightly between builds,
        # so exact search keeps the reported numbers reproducible.
        vector = self.model.encode(query, normalize_embeddings=True).astype(np.float32)
        scores = self.embeddings @ vector
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [(self.doc_ids[index], float(scores[index])) for index in top]


def rrf(lists: list[list[tuple[str, float]]], k: int = 60, limit: int = DEPTH) -> list[tuple[str, float]]:
    from app.retrieval.hybrid import reciprocal_rank_fusion

    fused = reciprocal_rank_fusion(
        {f"list{i}": [{"id": doc_id} for doc_id, _ in ranked] for i, ranked in enumerate(lists)}, k=k, limit=limit
    )
    return [(item["id"], item["score"]) for item in fused]


class CohereReranker:
    """Cohere re-ranking with an on-disk cache so each query is only ever billed once."""

    def __init__(self, cache_name: str):
        import cohere

        settings = get_settings()
        self.client = cohere.Client(settings.cohere_api_key)
        self.model = settings.cohere_rerank_model
        self.path = CACHE_DIR / f"rerank_{cache_name}.json"
        self.cache = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.calls = 0
        self.last_call = 0.0

    def rerank(self, query_id: str, query: str, candidates: list[tuple[str, float]], texts: dict[str, str]):
        candidate_ids = [doc_id for doc_id, _ in candidates]
        key = f"{query_id}|{hashlib.sha1('|'.join(candidate_ids).encode()).hexdigest()[:12]}"
        legacy = self.cache.get(query_id)
        if key not in self.cache and legacy and sorted(doc_id for doc_id, _ in legacy["ranking"]) == sorted(candidate_ids):
            self.cache[key] = legacy  # same candidates as an earlier run: reuse the paid-for ranking
        if key not in self.cache:
            # trial keys allow 10 rerank calls per minute
            wait = 6.5 - (time.time() - self.last_call)
            if wait > 0:
                time.sleep(wait)
            for attempt in range(6):
                try:
                    start = time.perf_counter()
                    response = self.client.rerank(
                        query=query,
                        documents=[texts[doc_id] for doc_id, _ in candidates],
                        top_n=len(candidates),
                        model=self.model,
                    )
                    latency = time.perf_counter() - start
                    break
                except Exception as error:  # noqa: BLE001
                    transient = ("429", "500", "502", "503", "504", "TooManyRequests", "Connect", "Timeout", "ServiceUnavailable")
                    if not any(marker in str(error) or marker in type(error).__name__ for marker in transient):
                        raise
                    time.sleep(15 * (attempt + 1))
            else:
                raise RuntimeError("Cohere rate limit did not clear")
            self.last_call = time.time()
            self.calls += 1
            self.cache[key] = {
                "ranking": [[candidates[item.index][0], float(item.relevance_score)] for item in response.results],
                "latency": latency,
            }
            self.path.write_text(json.dumps(self.cache))
        entry = self.cache[key]
        return [tuple(pair) for pair in entry["ranking"]], entry["latency"]


# ---------------------------------------------------------------- metrics

# hyphens become spaces so "external-combustion" matches "external combustion" (applied to every method alike)
_UNICODE_DASHES = str.maketrans({"-": " ", "\u2010": " ", "\u2011": " ", "\u2012": " ", "\u2013": " ", "\u2014": " ", "\u202f": " ", "\u00a0": " "})


def squad_normalize(text: str) -> str:
    text = text.translate(_UNICODE_DASHES).lower()
    text = "".join(character for character in text if character not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, answers: list[str]) -> float:
    return float(any(squad_normalize(prediction) == squad_normalize(answer) for answer in answers))


def token_f1(prediction: str, answers: list[str]) -> float:
    best = 0.0
    prediction_tokens = squad_normalize(prediction).split()
    for answer in answers:
        answer_tokens = squad_normalize(answer).split()
        common = Counter(prediction_tokens) & Counter(answer_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(prediction_tokens)
        recall = overlap / len(answer_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def contains_answer(prediction: str, answers: list[str]) -> float:
    normalized = squad_normalize(prediction)
    return float(any(squad_normalize(answer) in normalized for answer in answers))


def save_json(name: str, payload) -> Path:
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2))
    return path
