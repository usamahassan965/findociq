"""Run the ingestion + indexing pipeline over a directory of PDFs.

Usage: python scripts/ingest.py data/documents/

GPU-poor? Run this script on a free Google Colab T4 with QDRANT_URL /
QDRANT_API_KEY pointing at a Qdrant Cloud free cluster - then query
locally on CPU.
"""

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from findociq.indexing.text_embedder import TextEmbedder
from findociq.indexing.vector_store import VectorStore
from findociq.indexing.visual_embedder import VisualEmbedder
from findociq.ingestion.pdf_processor import process_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="Directory containing PDFs")
    parser.add_argument("--batch-size", type=int, default=8, help="Pages per indexing batch")
    args = parser.parse_args()

    print("Stage 1/3: rendering pages and extracting text...")
    records = process_directory(args.directory)
    print(f"  {len(records)} pages from {len({r.doc_name for r in records})} documents")

    print("Stage 2/3: loading embedding models...")
    visual = VisualEmbedder()
    text = TextEmbedder()
    store = VectorStore()
    store.ensure_collections(text_dim=text.dim)
    print(f"  visual model on {visual.device}")

    print("Stage 3/3: embedding and upserting...")
    for start in tqdm(range(0, len(records), args.batch_size), desc="indexing"):
        batch = records[start : start + args.batch_size]
        images = [Image.open(r.image_path) for r in batch]
        visual_embeddings = visual.embed_images(images)
        text_embeddings = text.embed_texts([r.text or " " for r in batch])
        store.upsert_pages(batch, visual_embeddings, text_embeddings)

    print("Done. Start the API or UI and ask questions.")


if __name__ == "__main__":
    main()
