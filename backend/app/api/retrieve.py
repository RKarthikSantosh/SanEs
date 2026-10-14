from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.llm.answer import generate_answer
from app.retrieval.hybrid import hybrid_search

router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    top_n: Optional[int] = None


@router.post("/query")
def query_documents(request: QueryRequest):
    """Hybrid retrieval only: BM25 + vector search, fused with RRF and re-ranked."""
    return hybrid_search(request.question, top_n=request.top_n)


@router.post("/ask")
def ask(request: QueryRequest):
    """Full pipeline: hybrid retrieval followed by a citation-grounded Groq answer."""
    passages = hybrid_search(request.question, top_n=request.top_n)
    result = generate_answer(request.question, passages)
    result["passages"] = passages
    return result
