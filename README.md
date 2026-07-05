# FinDocIQ — Multimodal Financial Document Intelligence

Ask questions over annual reports, broker research, and economic outlooks — **including the
charts, tables, and figures that text-only RAG can't see**.

FinDocIQ retrieves document *page images* directly using ColQwen2.5 late-interaction visual
embeddings (the ColPali approach), fuses them with classic dense text retrieval, and answers
with a vision-language model that cites the exact pages it used.

## Why this is different from a typical RAG project

Text-only RAG pipelines OCR a PDF, chunk the text, and lose every chart, table layout, and
figure in the process. Financial documents are *dominated* by visual content — a revenue
trend chart or a segmented results table carries information that never survives OCR.
FinDocIQ treats each page as an image and retrieves it visually, so the VLM answers while
actually *looking* at the chart.

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              1. INGESTION                   │
   PDF documents  ───►  │  pypdfium2: page → PNG image + raw text     │
                        │  metadata: doc name, page number            │
                        └──────────────────┬──────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │              2. DUAL INDEXING               │
                        │  Visual: ColQwen2.5 multi-vector (128-d     │
                        │          per patch, late interaction)       │
                        │  Text:   BGE-small dense embeddings         │
                        │  Store:  Qdrant (MAX_SIM multivector +      │
                        │          cosine dense collections)          │
                        └──────────────────┬──────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │           3. HYBRID RETRIEVAL               │
                        │  Query → visual search + text search        │
                        │  Reciprocal Rank Fusion (RRF)               │
                        └──────────────────┬──────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │           4. VLM GENERATION                 │
                        │  Top-k page images → vision LLM             │
                        │  (Gemini free tier / Ollama / Claude)       │
                        │  Answer with page-level citations           │
                        └──────────────────┬──────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │            5. EVALUATION                    │
                        │  Retrieval: hit@k, MRR on QA dataset        │
                        │  Generation: LLM-judge faithfulness +       │
                        │  relevance scoring                          │
                        └──────────────────┬──────────────────────────┘
                                           │
                        ┌──────────────────▼──────────────────────────┐
                        │            6. SERVING                       │
                        │  FastAPI REST API + Streamlit chat UI       │
                        │  Docker Compose (API + Qdrant + UI)         │
                        └─────────────────────────────────────────────┘
```

## Free-tier stack (no paid services required)

| Stage | Component | Free option |
|---|---|---|
| Ingestion | pypdfium2 | Open source, local |
| Visual embeddings | ColQwen2.5 (`vidore/colqwen2.5-v0.2`) | Open weights; runs locally on GPU, or index on a free Google Colab T4 |
| Text embeddings | `BAAI/bge-small-en-v1.5` | Open weights, runs on CPU |
| Vector DB | Qdrant | Free local Docker, or Qdrant Cloud free 1 GB cluster |
| Generation (VLM) | Gemini 2.5 Flash | Google AI Studio free tier (generous daily quota) |
| Generation (local) | Qwen2.5-VL via Ollama | Fully local, free |
| Generation (premium) | Claude (Anthropic API) | Optional, paid — highest quality |
| Evaluation | LLM-judge via same free VLM | Free |
| API + UI | FastAPI + Streamlit | Open source |
| Hosting demo | Hugging Face Spaces / Streamlit Community Cloud | Free |

> **GPU note:** ColQwen2.5 embedding is the only GPU-hungry stage. Without a local GPU,
> run `scripts/ingest.py` once on a free Colab T4 notebook and point it at a Qdrant Cloud
> free cluster — queries and the UI then run fine on CPU (query embedding is a single
> short text, fast even on CPU).

## Quickstart

```bash
# 1. Install (Python 3.11+)
pip install -e .

# 2. Start Qdrant
docker compose up -d qdrant

# 3. Configure — copy and fill in your free Gemini API key from aistudio.google.com
cp .env.example .env

# 4. Ingest documents (put PDFs in data/documents/ first)
python scripts/ingest.py data/documents/

# 5. Ask questions — API
uvicorn findociq.api.main:app --reload
# or UI
streamlit run ui/app.py

# 6. Evaluate
python scripts/evaluate.py data/eval/eval_questions.json
```

## Evaluation

`scripts/evaluate.py` runs the QA dataset in `data/eval/` through the full pipeline and reports:

- **Retrieval**: hit@1, hit@5, MRR against annotated gold pages
- **Generation**: LLM-judge scores for faithfulness (is the answer grounded in the
  retrieved pages?) and relevance (does it answer the question?)

### Results

10-question gold-annotated set over a 75-page corpus (Berkshire Hathaway 2024
shareholder letter + EIA Short-Term Energy Outlook, June 2026), spanning prose,
tables, and chart pages. Pages embedded with ColQwen2.5 on GPU, hybrid retrieval
against Qdrant, answers generated by Gemini 2.5 Flash, judged by Gemini 2.5
Flash-Lite (a different model as judge reduces self-grading bias).

| Metric | Score |
|---|---|
| hit@1 | 0.80 |
| hit@5 | 0.90 |
| MRR | 0.83 |
| Faithfulness (LLM-judge, 1–5) | 5.0 |
| Relevance (LLM-judge, 1–5) | 5.0 |

Both hit@1 misses are dense quarterly-statistics-table lookups (e.g. "WTI spot
price for Q2 2026") where the exact table page ranked below related overview
pages — the classic hard case for page-level retrieval. Per-question details:
[`results/eval_results.json`](results/eval_results.json).

## Project layout

```
src/findociq/
├── config.py                  # pydantic-settings, all knobs in .env
├── ingestion/pdf_processor.py # PDF → page images + text
├── indexing/
│   ├── visual_embedder.py     # ColQwen2.5 late-interaction embeddings
│   ├── text_embedder.py       # BGE-small dense embeddings
│   └── vector_store.py        # Qdrant multivector + dense collections
├── retrieval/hybrid_retriever.py  # visual + text search, RRF fusion
├── generation/vlm_client.py   # Gemini / Ollama / Claude, one interface
├── evaluation/evaluate.py     # hit@k, MRR, LLM-judge
└── api/main.py                # FastAPI endpoints
ui/app.py                      # Streamlit chat with page-image display
scripts/                       # ingest.py, evaluate.py
```

## License

MIT
