import asyncio
import uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, File, HTTPException, UploadFile

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.embeddings.text_embeddings import embed_texts
from app.loaders.document_loader import SUPPORTED_EXTENSIONS, extract_pages
from app.retrieval.bm25_store import invalidate_index
from app.retrieval.chroma_store import store_chunks

router = APIRouter()

# in-memory task tracker: task_id -> status dict
_tasks: dict[str, dict] = {}

# thread pool for CPU-bound processing work
_executor = ThreadPoolExecutor(max_workers=2)


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


def _process_document(task_id: str, file_path: Path, filename: str):
    """Heavy processing that runs in a background thread."""
    task = _tasks[task_id]
    try:
        # Extract pages
        task["step"] = "extracting"
        task["message"] = "Extracting text from document..."
        pages = extract_pages(file_path)

        # Chunking
        task["step"] = "chunking"
        task["message"] = f"Splitting into chunks ({len(pages)} pages found)..."
        chunks = chunk_pages(pages, filename)
        if not chunks:
            task["status"] = "failed"
            task["step"] = "error"
            task["message"] = "No text could be extracted from the document"
            return

        # Embedding
        task["step"] = "embedding"
        task["message"] = f"Generating embeddings for {len(chunks)} chunks..."
        embeddings = embed_texts([chunk["text"] for chunk in chunks])
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding

        # Storing
        task["step"] = "storing"
        task["message"] = "Storing in vector database..."
        store_chunks(chunks)
        invalidate_index()

        # Done
        task["status"] = "completed"
        task["step"] = "done"
        task["message"] = "Processing complete"
        task["result"] = {
            "filename": filename,
            "pages": len(pages),
            "text_length": sum(len(page_text) for page_text in pages),
            "chunks": len(chunks),
            "preview": [{"chunk_id": c["chunk_id"], "page": c["page"], "text": c["text"][:200]} for c in chunks[:3]],
        }
    except Exception as e:
        task["status"] = "failed"
        task["step"] = "error"
        task["message"] = str(e)


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # validate file type
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    # save file to disk
    raw_dir = Path(get_settings().raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    file_path = raw_dir / filename
    file_path.write_bytes(await file.read())

    # create a background task for the heavy processing
    task_id = uuid.uuid4().hex[:12]
    _tasks[task_id] = {
        "status": "processing",
        "step": "queued",
        "message": "Upload received, starting processing...",
        "filename": filename,
        "result": None,
    }

    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _process_document, task_id, file_path, filename)

    return {
        "task_id": task_id,
        "status": "processing",
        "filename": filename,
    }


@router.get("/upload/status/{task_id}")
async def upload_status(task_id: str):
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
