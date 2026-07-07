"""
Image extraction and processing strategies for multimodal RAG.

The strategy pattern here is the research variable:
  - CaptionOnlyStrategy : no LLM call, uses existing caption text only
  - ClaudeVisionStrategy: calls Claude Vision to generate rich description (default)

To add a new strategy (e.g. CLIP embedding), subclass ImageStrategy.
"""

import base64
from abc import ABC, abstractmethod
from pathlib import Path

import fitz  # pymupdf

MAX_IMAGE_BYTES = 8 * 1024 * 1024  # 8 MB — stay safely under Claude's 10 MB limit


# ---------------------------------------------------------------------------
# Pluggable strategy interface
# ---------------------------------------------------------------------------

class ImageStrategy(ABC):
    @abstractmethod
    def describe(self, image_bytes: bytes) -> str:
        """Given raw PNG bytes, return a text description."""


class CaptionOnlyStrategy(ImageStrategy):
    """Returns empty string — relies solely on adaptive-chunking's caption extraction."""

    def describe(self, image_bytes: bytes) -> str:
        return ""


class ClaudeVisionStrategy(ImageStrategy):
    """Calls Claude Vision API to generate a rich description of the image."""

    PROMPT = (
        "Describe this image concisely but completely. "
        "If it contains a chart or graph, describe the data, axes, and key trends. "
        "If it contains a diagram or figure, describe the structure and key elements. "
        "If it contains a table, summarize the key data. "
        "If it is decorative or a logo, say so briefly."
    )

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def describe(self, image_bytes: bytes) -> str:
        # detect format from magic bytes so JPEG and PNG both work
        media_type = "image/jpeg" if image_bytes[:3] == b'\xff\xd8\xff' else "image/png"
        b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        message = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": media_type, "data": b64},
                    },
                    {"type": "text", "text": self.PROMPT},
                ],
            }],
        )
        return message.content[0].text


# ---------------------------------------------------------------------------
# Image extraction from PDF
# ---------------------------------------------------------------------------

def extract_images_from_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    min_width: int = 100,
    min_height: int = 100,
) -> list[dict]:
    """
    Extract unique images from a PDF, save as PNG files.

    Returns a list of dicts with keys:
      page_num, image_path, image_bytes, width, height
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    images = []
    seen_xrefs: set[int] = set()

    for page_num, page in enumerate(doc):
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)

            base_image = doc.extract_image(xref)
            width, height = base_image["width"], base_image["height"]

            if width < min_width or height < min_height:
                continue

            pix = fitz.Pixmap(doc, xref)
            if pix.n > 4:  # convert CMYK to RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)

            # use PNG by default; fall back to JPEG if over size limit
            image_bytes = pix.tobytes("png")
            ext = "png"
            if len(image_bytes) > MAX_IMAGE_BYTES:
                image_bytes = pix.tobytes("jpeg")
                ext = "jpg"

            img_filename = f"{pdf_path.stem}_p{page_num + 1}_img{img_index}.{ext}"
            img_path = output_dir / img_filename
            with open(img_path, "wb") as f:
                f.write(image_bytes)

            images.append({
                "page_num": page_num + 1,
                "image_path": str(img_path),
                "image_bytes": image_bytes,
                "width": width,
                "height": height,
            })

    doc.close()
    return images


# ---------------------------------------------------------------------------
# Docling-based figure extraction (handles both raster and vector figures)
# ---------------------------------------------------------------------------

def extract_figures_with_docling(
    pdf_path: str | Path,
    output_dir: str | Path,
    strategy: ImageStrategy,
    doc_name: str | None = None,
    chunk_index_start: int = 0,
    min_width: int = 100,
    min_height: int = 100,
) -> list[dict]:
    """
    Use Docling to detect all figure regions (raster + vector), render each
    region with PyMuPDF at 2x resolution, then describe with the strategy.

    This replaces the old get_images() approach which missed vector graphics.
    """
    from docling.document_converter import DocumentConverter
    from docling_core.types.doc import PictureItem

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if doc_name is None:
        doc_name = pdf_path.stem

    # Docling pass: detect figure locations
    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    dl_doc = result.document

    # PyMuPDF: render figure regions
    fitz_doc = fitz.open(str(pdf_path))
    chunks = []
    fig_idx = 0

    for item, _ in dl_doc.iterate_items():
        if not isinstance(item, PictureItem):
            continue
        if not item.prov:
            continue

        prov = item.prov[0]
        page_no = prov.page_no  # 1-indexed
        bbox = prov.bbox

        fitz_page = fitz_doc[page_no - 1]
        page_h = fitz_page.rect.height

        # Docling bbox is bottom-left origin; convert to PyMuPDF top-left origin
        try:
            tl = bbox.to_top_left_origin(page_h)
            clip = fitz.Rect(tl.l, tl.t, tl.r, tl.b)
        except AttributeError:
            # fallback: manual conversion
            clip = fitz.Rect(bbox.l, page_h - bbox.t, bbox.r, page_h - bbox.b)

        # render at 2x resolution
        pix = fitz_page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip)

        if pix.width < min_width or pix.height < min_height:
            continue

        # compress if needed
        image_bytes = pix.tobytes("png")
        ext = "png"
        if len(image_bytes) > MAX_IMAGE_BYTES:
            image_bytes = pix.tobytes("jpeg")
            ext = "jpg"

        img_path = output_dir / f"{pdf_path.stem}_p{page_no}_fig{fig_idx}.{ext}"
        img_path.write_bytes(image_bytes)

        description = strategy.describe(image_bytes)
        fig_idx += 1

        if not description.strip():
            continue

        chunks.append({
            "doc_name": doc_name,
            "chunk_index": chunk_index_start + fig_idx,
            "chunk_text": description,
            "chunk_pages": [page_no],
            "titles_context": "",
            "chunk_len": len(description.split()),
            "type": "figure",
            "image_path": str(img_path),
        })

    fitz_doc.close()
    return chunks
