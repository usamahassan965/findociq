"""Stage 6 - Serving: FastAPI REST API.

POST /query  -> answer + cited pages
GET  /health -> liveness + configured provider
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from findociq.config import get_settings
from findociq.generation.vlm_client import get_vlm_client
from findociq.retrieval.hybrid_retriever import HybridRetriever

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models once at startup, not per request
    state["retriever"] = HybridRetriever()
    state["vlm"] = get_vlm_client()
    yield
    state.clear()


app = FastAPI(title="FinDocIQ", version="0.1.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None


class PageCitation(BaseModel):
    page_id: str
    doc_name: str
    page_number: int
    image_path: str
    fused_score: float
    sources: list[str]


class QueryResponse(BaseModel):
    answer: str
    pages: list[PageCitation]


@app.get("/health")
def health():
    settings = get_settings()
    return {"status": "ok", "provider": settings.vlm_provider}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    pages = state["retriever"].retrieve(request.question, top_k=request.top_k)
    answer = state["vlm"].answer(request.question, pages)
    return QueryResponse(
        answer=answer,
        pages=[
            PageCitation(
                page_id=p.page_id,
                doc_name=p.doc_name,
                page_number=p.page_number,
                image_path=p.image_path,
                fused_score=p.fused_score,
                sources=p.sources,
            )
            for p in pages
        ],
    )
