"""Stage 1 - Ingestion: render PDF pages to images and extract per-page text.

Each page becomes one retrieval unit carrying both representations:
the rendered PNG (for visual embedding) and the raw text layer (for dense
text embedding). Page images are persisted to disk so the UI and the VLM
can load them at answer time without re-rendering.
"""

from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from findociq.config import get_settings


@dataclass
class PageRecord:
    doc_name: str
    page_number: int  # 1-indexed
    image_path: Path
    text: str

    @property
    def page_id(self) -> str:
        return f"{self.doc_name}::p{self.page_number}"


def process_pdf(pdf_path: Path) -> list[PageRecord]:
    settings = get_settings()
    doc_name = pdf_path.stem
    out_dir = settings.pages_dir / doc_name
    out_dir.mkdir(parents=True, exist_ok=True)

    pdf = pdfium.PdfDocument(str(pdf_path))
    records: list[PageRecord] = []
    scale = settings.render_dpi / 72.0

    try:
        for i, page in enumerate(pdf):
            page_number = i + 1
            image_path = out_dir / f"page_{page_number:04d}.png"

            if not image_path.exists():
                bitmap = page.render(scale=scale)
                image: Image.Image = bitmap.to_pil()
                image.save(image_path, format="PNG")

            textpage = page.get_textpage()
            text = textpage.get_text_bounded() or ""
            textpage.close()

            records.append(
                PageRecord(
                    doc_name=doc_name,
                    page_number=page_number,
                    image_path=image_path,
                    text=text.strip(),
                )
            )
    finally:
        pdf.close()

    return records


def process_directory(directory: Path) -> list[PageRecord]:
    records: list[PageRecord] = []
    pdfs = sorted(directory.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {directory}")
    for pdf_path in pdfs:
        records.extend(process_pdf(pdf_path))
    return records
