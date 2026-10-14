from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.embeddings.text_embeddings import embed_texts
from app.loaders.document_loader import SUPPORTED_EXTENSIONS, extract_pages
from app.retrieval.bm25_store import invalidate_index
from app.retrieval.chroma_store import store_chunks

router = APIRouter()


def chunk_pages(pages: list[str], filename: str) -> list[dict]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)

    chunks = []
    for page_number, page_text in enumerate(pages, start=1):
        for chunk_text in splitter.split_text(page_text):
            chunks.append(
                {
                    "chunk_id": len(chunks) + 1,
                    "text": chunk_text,
                    "page": page_number,
                    "filename": filename,
                }
            )
    return chunks


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # keep only the base name so a crafted filename cannot write outside data/raw
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    raw_dir = Path(get_settings().raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / filename
    file_path.write_bytes(await file.read())

    try:
        pages = extract_pages(file_path)
    except Exception as error:
        raise HTTPException(status_code=422, detail=f"Could not read document: {error}") from error

    chunks = chunk_pages(pages, filename)
    if not chunks:
        raise HTTPException(status_code=422, detail="No text could be extracted from the document")

    for chunk, embedding in zip(chunks, embed_texts([chunk["text"] for chunk in chunks])):
        chunk["embedding"] = embedding
    store_chunks(chunks)
    invalidate_index()

    return {
        "filename": filename,
        "pages": len(pages),
        "text_length": sum(len(page_text) for page_text in pages),
        "chunks": len(chunks),
        "preview": [{"chunk_id": c["chunk_id"], "page": c["page"], "text": c["text"][:200]} for c in chunks[:3]],
    }
