"""End-to-end RAG comparison on SQuAD v1.1 dev: retrieval -> Groq answer -> answer quality + RAGAS.

Usage (from backend/):
  .venv/bin/python -m experiments.run_e2e generate
  .venv/bin/python -m experiments.run_e2e judge
  .venv/bin/python -m experiments.run_e2e report
"""
import argparse
import json
import re
import statistics
import time

import numpy as np

from experiments.common import (
    CACHE_DIR,
    CohereReranker,
    Retrievers,
    contains_answer,
    load_squad,
    rrf,
    save_json,
    token_f1,
)

SAMPLE_SIZE = 60  # questions drawn with seed 42
USED = 45  # questions answered by every method before Groq's free daily token quota (200k per model) ran out
CONTEXT_K = 5
JUDGED = ["bm25", "dense", "hybrid_rerank"]  # the judge model's daily quota covers three methods
CONFIGS = ["bm25", "dense", "hybrid", "hybrid_rerank"]
GENERATIONS = CACHE_DIR / "e2e_generations.json"
_CITATION = re.compile(r"\s*\[\d+\]")


def _load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def generate() -> None:
    from app.config import get_settings
    from app.llm.answer import generate_answer

    corpus, sample = load_squad(SAMPLE_SIZE)
    sample = sample[:USED]
    print(f"SQuAD corpus: {len(corpus)} paragraphs, {len(sample)} sampled questions", flush=True)
    retrievers = Retrievers("squad_dev", corpus)
    reranker = CohereReranker("squad")
    cache = _load(GENERATIONS)

    for count, item in enumerate(sample, start=1):
        question = item["question"]
        bm25 = retrievers.bm25_search(question)
        dense = retrievers.dense_search(question)
        hybrid = rrf([dense, bm25])
        reranked, _ = reranker.rerank(item["id"], question, hybrid, corpus)
        rankings = {"bm25": bm25, "dense": dense, "hybrid": hybrid, "hybrid_rerank": reranked}

        for config, ranked in rankings.items():
            key = f"{config}|{item['id']}"
            if key in cache:
                continue
            top = [doc_id for doc_id, _ in ranked[:CONTEXT_K]]
            passages = [{"id": doc_id, "text": corpus[doc_id], "metadata": {"filename": doc_id}} for doc_id in top]
            start = time.perf_counter()
            result = generate_answer(question, passages)
            cache[key] = {
                "question": question,
                "answers": item["answers"],
                "gold_doc": item["gold_doc"],
                "ranking": [doc_id for doc_id, _ in ranked],
                "contexts": [corpus[doc_id] for doc_id in top],
                "response": result["answer"],
                "cited_ids": [citation["id"] for citation in result["citations"]],
                "abstained": result["abstained"],
                "generation_seconds": time.perf_counter() - start,
            }
            GENERATIONS.write_text(json.dumps(cache))
        print(f"  {count}/{len(sample)} questions ({reranker.calls} new Cohere calls)", flush=True)
    print(f"model: {get_settings().groq_model}")


JUDGE_PROMPT = """Context passages:
{contexts}

Answer:
{answer}

Split the answer into its individual factual claims. For each claim, decide whether it is fully supported by
the context passages (true) or not (false). Reply with JSON only, in this form:
{{"claims": [{{"claim": "...", "supported": true}}]}}"""


def _parse_claims(text: str):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        return None
    try:
        claims = json.loads(text[start : end + 1]).get("claims", [])
    except json.JSONDecodeError:
        return None
    return [bool(claim.get("supported")) for claim in claims if isinstance(claim, dict)]


def judge(configs: list[str]) -> None:
    """RAGAS faithfulness (supported claims / all claims) with one judge call per answer.

    The RAGAS library makes two LLM calls per answer; one combined call keeps the experiment inside
    Groq's free daily token quota. The CI gate (experiments/ragas_regression.py) uses the library itself.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.config import get_settings
    from app.llm.groq_llm import invoke_with_fallback

    model = get_settings().groq_judge_model
    generations = _load(GENERATIONS)
    for config in [config for config in configs if config in JUDGED]:
        path = CACHE_DIR / f"e2e_faithfulness_{config}.json"
        scores = _load(path)
        keys = [key for key in generations if key.startswith(f"{config}|") and key not in scores and not generations[key]["abstained"]]
        print(f"{config}: {len(keys)} answers to judge", flush=True)
        for key in keys:
            row = generations[key]
            contexts = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(row["contexts"], start=1))
            messages = [
                SystemMessage(content="You are a strict fact-checking judge."),
                HumanMessage(content=JUDGE_PROMPT.format(contexts=contexts, answer=_CITATION.sub("", row["response"]))),
            ]
            verdicts = _parse_claims(invoke_with_fallback(messages, model=model))
            scores[key] = {"faithfulness": (sum(verdicts) / len(verdicts)) if verdicts else None, "claims": len(verdicts or [])}
            path.write_text(json.dumps(scores))
        print(f"  {config}: {len(scores)} judged", flush=True)


def _mean(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def report() -> None:
    from scipy import stats

    generations = _load(GENERATIONS)
    _, sample = load_squad(SAMPLE_SIZE)
    used_ids = {item["id"] for item in sample[:USED]}
    summary, per_question = {}, {}
    for config in CONFIGS:
        rows = {
            key.split("|", 1)[1]: value
            for key, value in generations.items()
            if key.startswith(f"{config}|") and key.split("|", 1)[1] in used_ids
        }
        if not rows:
            continue
        judged_scores = _load(CACHE_DIR / f"e2e_faithfulness_{config}.json")
        hit1, hit5, rr, accuracy, f1, cited_gold, abstained, faith, seconds = ([] for _ in range(9))
        retrieval_errors = generation_errors = 0
        for question_id, row in sorted(rows.items()):
            ranking = row["ranking"]
            gold = row["gold_doc"]
            rank = ranking.index(gold) + 1 if gold in ranking[:CONTEXT_K] else None
            hit1.append(float(rank == 1))
            hit5.append(float(rank is not None))
            rr.append(1.0 / rank if rank else 0.0)
            response = _CITATION.sub("", row["response"])
            accuracy.append(contains_answer(response, row["answers"]))
            if not accuracy[-1]:
                if rank is None:
                    retrieval_errors += 1
                else:
                    generation_errors += 1
            f1.append(token_f1(response, row["answers"]))
            cited_gold.append(float(gold in row["cited_ids"]))
            abstained.append(float(row["abstained"]))
            faith.append(judged_scores.get(f"{config}|{question_id}", {}).get("faithfulness"))
            seconds.append(row["generation_seconds"])
        per_question[config] = {"accuracy": np.array(accuracy), "faithfulness": faith}
        summary[config] = {
            "questions": len(rows),
            "Hit@1": _mean(hit1),
            "Hit@5": _mean(hit5),
            "MRR@5": _mean(rr),
            "AnswerAccuracy": _mean(accuracy),
            "TokenF1": _mean(f1),
            "GoldCitationRate": _mean(cited_gold),
            "AbstentionRate": _mean(abstained),
            "Faithfulness": _mean(faith),
            "faithfulness_judged": sum(value is not None for value in faith),
            "RetrievalErrors": retrieval_errors,
            "GenerationErrors": generation_errors,
            "generation_seconds_mean": statistics.mean(seconds),
        }

    significance = {}
    for system in ("hybrid", "hybrid_rerank"):
        for baseline in ("bm25", "dense"):
            if system in per_question and baseline in per_question:
                a, b = per_question[system]["accuracy"], per_question[baseline]["accuracy"]
                # McNemar-style exact test on paired correct/incorrect outcomes
                better, worse = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
                p_value = stats.binomtest(better, better + worse).pvalue if better + worse else 1.0
                significance[f"{system}_vs_{baseline}"] = {"fixed": better, "broke": worse, "mcnemar_p": float(p_value)}

    payload = {"dataset": "squad-v1.1-dev", "sample": USED, "context_k": CONTEXT_K, "summary": summary, "significance": significance}
    path = save_json("e2e_squad.json", payload)
    for config, metrics in summary.items():
        print(config, {name: round(value, 4) if isinstance(value, float) else value for name, value in metrics.items()})
    print(significance)
    print("saved", path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["generate", "judge", "report"])
    parser.add_argument("--configs", default=",".join(CONFIGS))
    parser.add_argument("--key", type=int, default=0, help="unused; kept for run_all.sh compatibility")
    args = parser.parse_args()
    if args.phase == "generate":
        generate()
    elif args.phase == "judge":
        judge(args.configs.split(","))
    else:
        report()


if __name__ == "__main__":
    main()
