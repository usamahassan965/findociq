"""Stage 3 - Hybrid retrieval: visual + text search fused with RRF.

Reciprocal Rank Fusion needs no score normalization across the two very
different scoring scales (MaxSim sums vs cosine similarity), which is why
it is the standard choice for fusing heterogeneous retrievers.
"""

from dataclasses import dataclass

from findociq.config import get_settings
from findociq.indexing.text_embedder import TextEmbedder
from findociq.indexing.vector_store import VectorStore
from findociq.indexing.visual_embedder import VisualEmbedder


@dataclass
class RetrievedPage:
    page_id: str
    doc_name: str
    page_number: int
    image_path: str
    text: str
    fused_score: float
    sources: list[str]  # which retrievers surfaced it: "visual", "text"


class HybridRetriever:
    def __init__(
        self,
        store: VectorStore | None = None,
        visual: VisualEmbedder | None = None,
        text: TextEmbedder | None = None,
    ):
        self.settings = get_settings()
        self.store = store or VectorStore()
        self.visual = visual or VisualEmbedder()
        self.text = text or TextEmbedder()

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedPage]:
        top_k = top_k or self.settings.top_k
        # Over-fetch each path so fusion has candidates to work with
        fetch_k = max(top_k * 3, 15)

        visual_hits = self.store.search_visual(self.visual.embed_query(query), limit=fetch_k)
        text_hits = self.store.search_text(self.text.embed_query(query), limit=fetch_k)

        fused = self._rrf({"visual": visual_hits, "text": text_hits})
        return fused[:top_k]

    def _rrf(self, ranked_lists: dict[str, list[dict]]) -> list[RetrievedPage]:
        k = self.settings.rrf_k
        scores: dict[str, float] = {}
        meta: dict[str, dict] = {}
        sources: dict[str, list[str]] = {}

        for source, hits in ranked_lists.items():
            for rank, hit in enumerate(hits):
                pid = hit["page_id"]
                scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
                meta.setdefault(pid, hit)
                sources.setdefault(pid, []).append(source)

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return [
            RetrievedPage(
                page_id=pid,
                doc_name=meta[pid]["doc_name"],
                page_number=meta[pid]["page_number"],
                image_path=meta[pid]["image_path"],
                text=meta[pid].get("text", ""),
                fused_score=score,
                sources=sources[pid],
            )
            for pid, score in ordered
        ]
