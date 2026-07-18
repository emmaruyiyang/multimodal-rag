"""FastAPI backend for the multimodal RAG pipeline."""

import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "adaptive-chunking" / "src"))

load_dotenv(ROOT / ".env")

from src.pipeline import index_document
from src.pipeline import query as rag_query

app = FastAPI(title="Multimodal RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)


class QueryRequest(BaseModel):
    question: str
    doc_name: str
    top_k: int = 5


@app.get("/api/documents")
def list_documents():
    return [p.stem for p in sorted(DATA_DIR.glob("*.pdf"))]


@app.post("/api/documents/index")
def index_pdf(file: UploadFile = File(...)):
    pdf_path = DATA_DIR / file.filename
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        result = index_document(pdf_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/documents/query")
def query_doc(req: QueryRequest):
    try:
        return rag_query(req.question, doc_name=req.doc_name, top_k=req.top_k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/pdf/{doc_name}")
def get_pdf(doc_name: str):
    pdf_path = DATA_DIR / f"{doc_name}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(str(pdf_path), media_type="application/pdf")


# Serve frontend — must be mounted last so API routes take priority
app.mount("/", StaticFiles(directory=str(ROOT / "frontend"), html=True), name="frontend")
