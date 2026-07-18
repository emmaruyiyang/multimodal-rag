# Multimodal RAG

RAG pipeline for PDFs with pluggable image processing strategies — designed for comparing how different image handling approaches affect retrieval quality.

## How it works

**Offline** — run once per document:

```
PDF
 ├── Text + Tables ──► Docling (adaptive-chunking) ──► text chunks
 └── Images ──────────► [pluggable strategy] ─────────► description chunks
                                  │
                         both embedded with
                     OpenAI text-embedding-3-small
                                  │
                             stored in Qdrant
```

**Online** — runs per query:

```
question ──► embed ──► Qdrant search ──► top-k chunks ──► Claude ──► answer + page refs
```

The image processing strategy is the **research variable**: swapping it lets you measure how different approaches (caption-only vs. Vision LLM vs. CLIP etc.) affect retrieval recall.

Each chunk stored in Qdrant carries `doc_name`, `chunk_pages`, and `type` (`text` / `table` / `figure_caption` / `image`), enabling future page-jump features in a PDF reader frontend.

## Quick Start

**1. Start Qdrant**
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```
Dashboard: `http://localhost:6333/dashboard`

**2. Start the web app**
```bash
conda activate multimodal-rag
uvicorn api.main:app --reload
```
Open `http://localhost:8000` — upload a PDF, wait for indexing, then ask questions.

> PDF viewer requires an Adobe PDF Embed API Client ID — register at developer.adobe.com and set it in `frontend/app.js`. Add `localhost` as the allowed domain.

**3. Or use the Python API directly**
```python
from src.pipeline import index_document, query

index_document("data/my_paper.pdf")
result = query("What are the security features?", doc_name="my_paper")
print(result["answer"])
```

---

## Setup

```bash
pip install -r requirements.txt
pip install -e "adaptive-chunking[parsing]"
cp .env.example .env  # add ANTHROPIC_API_KEY and OPENAI_API_KEY
```

## Usage


## Swap image strategy

```python
from src.image_processor import CaptionOnlyStrategy, ClaudeVisionStrategy

index_document("data/my_paper.pdf", image_strategy=CaptionOnlyStrategy())   # fast, no LLM
index_document("data/my_paper.pdf", image_strategy=ClaudeVisionStrategy())  # default
```

To add a new strategy, subclass `ImageStrategy` in [src/image_processor.py](src/image_processor.py):

```python
from src.image_processor import ImageStrategy

class MyStrategy(ImageStrategy):
    def describe(self, image_bytes: bytes) -> str:
        return "..."
```
