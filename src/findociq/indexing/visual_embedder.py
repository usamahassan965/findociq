"""Stage 2a - Visual indexing: ColQwen2.5 late-interaction embeddings.

ColQwen2.5 (ColPali family) encodes a page image as a grid of patch tokens,
projecting each patch to a 128-d vector. Retrieval scores a query against a
page with MaxSim (late interaction): for every query token, take the best-
matching patch, then sum. This preserves spatial/layout information that a
single pooled vector destroys - which is exactly what charts and tables need.

Embedding pages needs a GPU to be pleasant (a free Colab T4 works: run
scripts/ingest.py there against a Qdrant Cloud free cluster). Embedding a
*query* is cheap and fine on CPU.
"""

import torch
from PIL import Image

from findociq.config import get_settings


def _pick_device() -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


class VisualEmbedder:
    def __init__(self, model_name: str | None = None):
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        settings = get_settings()
        self.model_name = model_name or settings.visual_model
        self.device, dtype = _pick_device()
        self.model = ColQwen2_5.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=self.device
        ).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(self.model_name)

    @torch.no_grad()
    def embed_images(self, images: list[Image.Image], batch_size: int = 4) -> list[list[list[float]]]:
        """Returns one multivector (n_patches x 128) per image."""
        all_embeddings: list[list[list[float]]] = []
        for start in range(0, len(images), batch_size):
            batch = self.processor.process_images(images[start : start + batch_size]).to(
                self.model.device
            )
            embeddings = self.model(**batch)
            for emb in torch.unbind(embeddings.to(torch.float32).cpu()):
                all_embeddings.append(emb.tolist())
        return all_embeddings

    @torch.no_grad()
    def embed_query(self, query: str) -> list[list[float]]:
        """Returns a multivector (n_tokens x 128) for the query text."""
        batch = self.processor.process_queries([query]).to(self.model.device)
        embeddings = self.model(**batch)
        return embeddings[0].to(torch.float32).cpu().tolist()
