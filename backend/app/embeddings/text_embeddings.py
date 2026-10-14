from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import get_settings


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().embedding_model)


def embed_text(text: str) -> list[float]:
    embedding = get_embedding_model().encode(text, normalize_embeddings=True)
    return embedding.tolist()


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    embeddings = get_embedding_model().encode(texts, batch_size=batch_size, normalize_embeddings=True)
    return embeddings.tolist()
