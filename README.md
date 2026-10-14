# SanEs: Hybrid-Retrieval Enterprise Document Intelligence Engine

Upload your documents and ask questions in plain language. SanEs finds the most relevant sections across your files and gives you a precise answer with references to the exact document and page it came from.

Supports PDF, DOCX, TXT, and Markdown files.

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- Groq API key (and optional fallback key)
- Cohere API key

---

## Backend

All backend commands run from the `backend/` directory.

### 1. Create and activate the virtual environment

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create `backend/.env` (gitignored):

```env
GROQ_API_KEY=your_groq_api_key
GROQ_API_KEY_FALLBACK=your_fallback_groq_key
COHERE_API_KEY=your_cohere_api_key

# Optional overrides (defaults shown)
GROQ_MODEL=openai/gpt-oss-20b
GROQ_JUDGE_MODEL=openai/gpt-oss-120b
GROQ_REASONING_EFFORT=low
COHERE_RERANK_MODEL=rerank-english-v3.0
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PATH=data/chroma
RAW_DIR=data/raw
```

### 4. Start the backend server

```bash
cd backend
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

API is available at `http://127.0.0.1:8000`. Interactive docs at `http://127.0.0.1:8000/docs`.

### Run tests

```bash
cd backend
source .venv/bin/activate
python -m pytest
```

---

## Frontend

All frontend commands run from the `frontend/` directory.

### 1. Install dependencies

```bash
cd frontend
npm install
```

### 2. Start the dev server

```bash
npm run dev
```

UI is available at `http://localhost:5173`. The backend must be running on port 8000.

---

## API Endpoints

| Method | Path      | Description                         |
| ------ | --------- | ----------------------------------- |
| `GET`  | `/health` | Backend liveness check              |
| `POST` | `/upload` | Upload a PDF, DOCX, TXT, or MD file |
| `POST` | `/ask`    | Hybrid retrieval + cited answer     |
| `POST` | `/query`  | Hybrid retrieval only (no LLM)      |

### Example: upload a document

```bash
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@your_document.pdf"
```

### Example: ask a question

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main finding?", "top_n": 5}'
```
