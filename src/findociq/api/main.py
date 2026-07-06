"""Stage 6 - Serving: FastAPI REST API.

POST /query         -> answer + cited pages + per-stage timings
POST /query/stream  -> plain-text token stream of the answer
GET  /health        -> liveness + configured provider
"""

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
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


class StageTimings(BaseModel):
    retrieval_ms: float
    generation_ms: float


class QueryResponse(BaseModel):
    answer: str
    pages: list[PageCitation]
    timings: StageTimings


@app.get("/health")
def health():
    settings = get_settings()
    return {"status": "ok", "provider": settings.vlm_provider}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    t0 = time.perf_counter()
    pages = state["retriever"].retrieve(request.question, top_k=request.top_k)
    t1 = time.perf_counter()
    answer = state["vlm"].answer(request.question, pages)
    t2 = time.perf_counter()
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
        timings=StageTimings(
            retrieval_ms=round((t1 - t0) * 1000, 1),
            generation_ms=round((t2 - t1) * 1000, 1),
        ),
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest):
    pages = state["retriever"].retrieve(request.question, top_k=request.top_k)
    return StreamingResponse(
        state["vlm"].answer_stream(request.question, pages),
        media_type="text/plain; charset=utf-8",
    )
