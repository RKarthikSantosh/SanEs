from functools import lru_cache

from chromadb import PersistentClient

from app.config import get_settings
from app.embeddings.text_embeddings import embed_text


COLLECTION_NAME = "document_chunks"


@lru_cache(maxsize=1)
def get_collection():
    client = PersistentClient(path=get_settings().chroma_path)
    # a larger search beam (ef_search) keeps HNSW results close to exact cosine search
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        configuration={"hnsw": {"space": "cosine", "ef_search": 100, "ef_construction": 200}},
    )


def store_chunks(chunks: list[dict]) -> None:
    if not chunks:
        return

    collection = get_collection()
    collection.upsert(
        ids=[f"{chunk['filename']}-{chunk['chunk_id']}" for chunk in chunks],
        documents=[chunk["text"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[
            {
                "chunk_id": chunk["chunk_id"],
                "page": chunk["page"],
                "filename": chunk["filename"],
            }
            for chunk in chunks
        ],
    )


def query_chunks(query: str, n_results: int = 3) -> dict:
    collection = get_collection()
    return collection.query(
        query_embeddings=[embed_text(query)],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )


def query_vectors(query: str, n_results: int = 5) -> list[dict]:
    results = query_chunks(query, n_results=n_results)
    return [
        {"id": chunk_id, "text": document, "metadata": metadata, "score": 1.0 - distance}
        for chunk_id, document, metadata, distance in zip(
            results.get("ids", [[]])[0],
            results.get("documents", [[]])[0],
            results.get("metadatas", [[]])[0],
            results.get("distances", [[]])[0],
        )
    ]
