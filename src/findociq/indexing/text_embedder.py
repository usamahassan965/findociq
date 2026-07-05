"""Stage 2b - Text indexing: dense embeddings of the page text layer.

BGE-small runs comfortably on CPU and complements visual retrieval: pages
that are mostly prose (management discussion, footnotes) match better on
text, while chart/table-heavy pages match better visually. The hybrid
retriever fuses both signals.
"""

from sentence_transformers import SentenceTransformer

from findociq.config import get_settings


class TextEmbedder:
    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.text_model
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()

    def embed_query(self, query: str) -> list[float]:
        # BGE models are trained with a query instruction prefix
        return self.model.encode(
            f"Represent this sentence for searching relevant passages: {query}",
            normalize_embeddings=True,
        ).tolist()
