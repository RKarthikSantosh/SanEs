import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.groq_llm import invoke_with_fallback

NOT_FOUND = "I could not find this in the provided documents."

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered context passages you are given.
Rules:
1. Every factual sentence must end with the number of the passage that supports it, in square brackets, e.g. [2] or [1][3].
2. Do not use any knowledge that is not in the passages. Do not guess.
3. If the passages do not contain the answer at all, reply with exactly: {NOT_FOUND}
   If they answer only part of the question, answer that part and say plainly which part the documents do not cover.
4. Keep the answer short and direct."""

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def build_context(passages: list[dict]) -> str:
    blocks = []
    for number, passage in enumerate(passages, start=1):
        metadata = passage.get("metadata") or {}
        source = metadata.get("filename", passage.get("id", "unknown"))
        page = metadata.get("page")
        label = f"{source}, page {page}" if page is not None else str(source)
        blocks.append(f"[{number}] ({label})\n{passage.get('text', '')}")
    return "\n\n".join(blocks)


def parse_citations(answer: str, passage_count: int) -> list[int]:
    cited = []
    for match in _CITATION_PATTERN.findall(answer):
        number = int(match)
        if 1 <= number <= passage_count and number not in cited:
            cited.append(number)
    return cited


def generate_answer(question: str, passages: list[dict], model: str | None = None) -> dict:
    if not passages:
        return {"answer": NOT_FOUND, "citations": [], "abstained": True}

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Context passages:\n\n{build_context(passages)}\n\nQuestion: {question}"),
    ]
    answer = invoke_with_fallback(messages, model=model).strip()
    cited = parse_citations(answer, len(passages))

    citations = []
    for number in cited:
        passage = passages[number - 1]
        metadata = passage.get("metadata") or {}
        score = passage.get("rerank_score")
        if score is None:
            score = passage.get("score")
        citations.append(
            {
                "number": number,
                "id": passage.get("id"),
                "filename": metadata.get("filename"),
                "page": metadata.get("page"),
                "text": passage.get("text", ""),
                "snippet": (passage.get("text") or "")[:300],
                "score": float(score) if score is not None else None,
            }
        )

    return {"answer": answer, "citations": citations, "abstained": answer.startswith(NOT_FOUND.rstrip("."))}
