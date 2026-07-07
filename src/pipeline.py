"""
End-to-end multimodal RAG pipeline.

Offline:  index_document(pdf_path)   — parse + embed + store (run once per doc)
Online:   query(question, doc_name)  — retrieve + generate  (runs per question)
"""

import sys
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

# make adaptive-chunking importable
sys.path.insert(0, str(Path(__file__).parent.parent / "adaptive-chunking" / "src"))

from .image_processor import ClaudeVisionStrategy, ImageStrategy, extract_figures_with_docling
from .indexer import COLLECTION, Indexer
from .retriever import Retriever

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
IMAGES_DIR = "./data/processed/images"


def _detect_chunk_type(chunk_text: str) -> str:
    if "<Table>" in chunk_text:
        return "table"
    if "<Figure>" in chunk_text:
        return "figure_caption"
    return "text"


def index_document(
    pdf_path: str | Path,
    image_strategy: ImageStrategy | None = None,
    qdrant_host: str = QDRANT_HOST,
    qdrant_port: int = QDRANT_PORT,
    collection: str = COLLECTION,
) -> dict:
    """
    Parse a PDF, extract images, embed everything, store in Qdrant.

    Args:
        pdf_path:       Path to the PDF file.
        image_strategy: How to process images. Defaults to ClaudeVisionStrategy.
        qdrant_path:    Where to persist Qdrant data.
        collection:     Qdrant collection name.

    Returns:
        Summary dict with doc_name, text_chunks, image_chunks counts.
    """
    from adaptive_chunking.parsing import DoclingParser
    from adaptive_chunking.pipeline import chunk_files

    pdf_path = Path(pdf_path)
    doc_name = pdf_path.stem

    if image_strategy is None:
        image_strategy = ClaudeVisionStrategy()

    print(f"[1/3] Parsing text and tables: {pdf_path.name}")
    parser = DoclingParser()
    text_chunks = chunk_files(pdf_path, parser=parser)
    for chunk in text_chunks:
        chunk["doc_name"] = doc_name  # override temp-dir-prefixed name
        chunk["type"] = _detect_chunk_type(chunk["chunk_text"])
    print(f"      {len(text_chunks)} chunks")

    print(f"[2/3] Extracting figures ({image_strategy.__class__.__name__})")
    images_dir = Path(IMAGES_DIR) / doc_name
    image_chunks = extract_figures_with_docling(
        pdf_path,
        output_dir=images_dir,
        strategy=image_strategy,
        doc_name=doc_name,
        chunk_index_start=len(text_chunks),
    )
    print(f"      {len(image_chunks)} figure chunks")

    all_chunks = text_chunks + image_chunks

    print(f"[3/3] Embedding and indexing {len(all_chunks)} chunks")
    indexer = Indexer(host=qdrant_host, port=qdrant_port)
    count = indexer.index_chunks(all_chunks, collection=collection)
    indexer.close()
    print(f"      Done — {count} points stored for '{doc_name}'")

    return {
        "doc_name": doc_name,
        "text_chunks": len(text_chunks),
        "image_chunks": len(image_chunks),
    }


def query(
    question: str,
    doc_name: str | None = None,
    top_k: int = 5,
    qdrant_host: str = QDRANT_HOST,
    qdrant_port: int = QDRANT_PORT,
    collection: str = COLLECTION,
) -> dict:
    """
    Retrieve relevant chunks and generate an answer with Claude.

    Args:
        question:   The user's question.
        doc_name:   Restrict search to this document (None = search all).
        top_k:      Number of chunks to retrieve.

    Returns:
        dict with 'answer' (str) and 'sources' (list of chunk summaries).
    """
    retriever = Retriever(host=qdrant_host, port=qdrant_port)
    chunks = retriever.search(question, doc_name=doc_name, top_k=top_k, collection=collection)

    if not chunks:
        return {"answer": "No relevant information found.", "sources": []}

    context_parts = []
    for c in chunks:
        pages = c.get("chunk_pages") or []
        page_str = f"Page {pages[0]}" if pages else "Unknown page"
        context_parts.append(f"[{page_str} | {c.get('type', 'text')}]\n{c['chunk_text']}")
    context = "\n\n---\n\n".join(context_parts)

    client = anthropic.Anthropic()
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Answer the question based on the document context below.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                f"Answer concisely and cite the page number(s) where the information comes from."
            ),
        }],
    )

    return {
        "answer": message.content[0].text,
        "sources": [
            {
                "type": c.get("type"),
                "pages": c.get("chunk_pages", []),
                "score": round(c.get("score", 0), 3),
                "preview": c["chunk_text"][:150] + "…" if len(c["chunk_text"]) > 150 else c["chunk_text"],
            }
            for c in chunks
        ],
    }
