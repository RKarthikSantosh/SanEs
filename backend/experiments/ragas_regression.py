"""CI answer-quality gate: run the full pipeline on a small fixed question set and score it with RAGAS.

Needs GROQ_API_KEY (Cohere is optional). Fails when mean faithfulness drops below the floor or the
system stops abstaining on an unanswerable question.
Usage: .venv/bin/python -m experiments.ragas_regression
"""
import re
import sys

from app.llm.answer import generate_answer
from experiments.common import Retrievers, rrf

FAITHFULNESS_FLOOR = 0.8

DOCUMENTS = {
    "handbook-1": "Employees receive 24 days of paid leave each calendar year. Unused leave of up to 10 days can be carried forward.",
    "handbook-2": "Travel claims must be filed within 30 days of the trip using form TR-7. Claims above INR 50,000 need director approval.",
    "handbook-3": "The IT helpdesk is open from 9 am to 6 pm on working days. Password resets are handled through the self-service portal.",
    "handbook-4": "All laptops must use full-disk encryption. Lost devices must be reported to security within 24 hours.",
    "handbook-5": "The office cafeteria serves lunch between 12:30 pm and 2:30 pm.",
}
QUESTIONS = [
    "How many days of paid leave do employees get?",
    "Which form is used for travel claims?",
    "Who must approve a travel claim above INR 50,000?",
    "How quickly must a lost laptop be reported?",
]
UNANSWERABLE = "What is the company's policy on remote work from abroad?"


def main() -> int:
    from app.config import get_settings
    import app.evaluation.ragas_compat  # noqa: F401
    from langchain_groq import ChatGroq
    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness

    retrievers = Retrievers("ci_handbook", DOCUMENTS)

    def answer(question):
        hybrid = rrf([retrievers.dense_search(question, 5), retrievers.bm25_search(question, 5)])[:3]
        passages = [{"id": doc_id, "text": DOCUMENTS[doc_id], "metadata": {"filename": doc_id}} for doc_id, _ in hybrid]
        return generate_answer(question, passages), passages

    rows = []
    for question in QUESTIONS:
        result, passages = answer(question)
        rows.append(
            {
                "user_input": question,
                "response": re.sub(r"\s*\[\d+\]", "", result["answer"]),
                "retrieved_contexts": [passage["text"] for passage in passages],
            }
        )

    settings = get_settings()
    llm = LangchainLLMWrapper(
        ChatGroq(model=settings.groq_judge_model, api_key=settings.groq_api_key, temperature=0, reasoning_effort="low", max_retries=8)
    )
    frame = evaluate(
        EvaluationDataset.from_list(rows), metrics=[Faithfulness(llm=llm)], run_config=RunConfig(max_workers=2), show_progress=False
    ).to_pandas()
    faithfulness = float(frame["faithfulness"].mean())
    abstained = answer(UNANSWERABLE)[0]["abstained"]
    print({"faithfulness": round(faithfulness, 3), "abstained_on_unanswerable": abstained})

    failures = []
    if faithfulness < FAITHFULNESS_FLOOR:
        failures.append(f"faithfulness {faithfulness:.3f} is below {FAITHFULNESS_FLOOR}")
    if not abstained:
        failures.append("the system answered a question the documents cannot answer")
    for failure in failures:
        print("REGRESSION:", failure)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
