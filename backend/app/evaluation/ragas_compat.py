import sys
import types

_MODULE = "langchain_community.chat_models.vertexai"

if _MODULE not in sys.modules:
    try:
        __import__(_MODULE)
    except ImportError:
        stub = types.ModuleType(_MODULE)

        class ChatVertexAI:
            pass

        stub.ChatVertexAI = ChatVertexAI
        sys.modules[_MODULE] = stub

import ragas

from langchain_core.embeddings import Embeddings


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain embeddings backed by the same sentence-transformer model the backend uses."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        from app.embeddings.text_embeddings import embed_texts

        return embed_texts(texts)

    def embed_query(self, text: str) -> list[float]:
        from app.embeddings.text_embeddings import embed_text

        return embed_text(text)


def ragas_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(SentenceTransformerEmbeddings())
