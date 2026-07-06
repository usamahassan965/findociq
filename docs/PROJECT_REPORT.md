# FinDocIQ — Engineering Case Study

*One-page project report: the problem, the options weighed at each stage, and real output from
every stage of the pipeline. All numbers below are actual captured artifacts, reproducible from
this repo — nothing is illustrative.*

---

## 1. The problem

Financial documents (annual reports, energy outlooks, broker research) are dominated by **charts
and tables**. A text-only RAG pipeline OCRs the PDF, chunks the text, and destroys exactly the
content the questions are about — a quarterly price table or a production-growth chart doesn't
survive chunking. The goal: a QA system that retrieves document **pages as images** and answers
while actually *looking* at them, under hard constraints:

- **No local GPU** and **zero budget** — every component free-tier or open-weights.
- Real-world corpus: Berkshire Hathaway 2024 shareholder letter (15 pages) + EIA Short-Term
  Energy Outlook, June 2026 (60 pages) — prose, dense tables, and chart pages.

## 2. Design decisions — options considered vs. chosen

| Stage | Options considered | Chosen | Why |
|---|---|---|---|
| Page rendering | OCR + chunking; Unstructured.io; **pypdfium2 page→PNG** | pypdfium2 | Keeps visual layout; also extracts raw text for the second lane; zero cost |
| Visual retrieval | CLIP page embeddings (single vector); **ColQwen2.5 late interaction (ColPali)**; caption-then-embed | ColQwen2.5 | Single-vector CLIP collapses a whole page into one point — late interaction keeps ~755 patch vectors per page, so a query token can match one table cell region |
| Text retrieval | BM25; OpenAI embeddings (paid); **BGE-small-en-v1.5** | BGE-small | Open weights, 384-d, runs on CPU; complements the visual lane on prose questions |
| Fusion | Score normalization + weighted sum; learned reranker; **Reciprocal Rank Fusion (k=60)** | RRF | MAX_SIM (~20–30) and cosine (~0.6–0.8) live on incomparable scales; RRF is scale-free and needs no training data |
| Vector DB | FAISS (no server, no payloads); pgvector; **Qdrant** | Qdrant | Native multivector MAX_SIM support + payload storage + free local Docker / 1 GB cloud tier |
| Generation | OCR text → text LLM; **page images → VLM (Gemini 2.5 Flash)**; Ollama qwen2.5-vl (local fallback) | Gemini VLM | The model answers from the rendered page, so table/chart questions work; free tier |
| Evaluation | Vibes; RAGAS; **hit@k / MRR on gold pages + LLM-judge** | Custom | Page-level gold labels make retrieval metrics exact; judge on a *different* model (Flash-Lite) to reduce self-grading bias |
| GPU for indexing | Colab (session limits); Modal (credit card); **Kaggle API kernels** | Kaggle | Scriptable push/status/output CLI → fully automated remote embedding job |

## 3. Pipeline walkthrough — real output at every stage

### Stage 1 — Ingestion (`pypdfium2`)
75 pages rendered to PNG (~990×1290 / 1020×1320 px) + raw text per page + metadata
(`doc_name`, `page_number`, `page_id` like `berkshire_2024_letter::p5`).

### Stage 2 — Dual indexing (ColQwen2.5 + BGE-small → Qdrant)
Embedded on a Kaggle P100 GPU kernel (fp16, batch size 1):

- **Visual lane:** ColQwen2.5 (`vidore/colqwen2.5-v0.2`) — 75 pages × **755 patch vectors ×
  128-d** each (late-interaction multivector, Qdrant MAX_SIM comparator) — ~19 MB of fp16
  artifacts shipped back, instead of the 7.5 GB model.
- **Text lane:** BGE-small (`BAAI/bge-small-en-v1.5`) — 75 × **384-d** dense vectors (cosine),
  embedded on CPU.
- One page (a sparse cover page) produced NaN patch vectors from fp16 overflow — sanitized with
  `nan_to_num` (zero vectors can never win MAX_SIM, so this is lossless for ranking).

### Stage 3 — Hybrid retrieval (the money stage)

Each lane returns its top-15 candidates (`fetch_k = 15`); Reciprocal Rank Fusion (`k = 60`)
merges the two ranked lists and keeps the top 5 (`top_k = 5`) for generation.

**Query:** *"What were Berkshire's insurance-underwriting earnings in 2024, and how do they
compare to 2023?"* — gold page **p5** (the earnings breakdown table).

| Rank | Visual lane (MAX_SIM) | Text lane (cosine) | RRF fused |
|---|---|---|---|
| 1 | **p5** (29.47) ✅ | p4 (0.778) ❌ | **p5** ✅ |
| 2 | p4 (25.42) | **p5** (0.767) | p4 |
| 3 | p10 (23.07) | p10 (0.688) | p10 |

The text lane ranks the prose page p4 first — the extracted text of the table page scores lower
than surrounding narrative. The **visual lane sees the table itself** and ranks p5 first with a
wide margin; fusion keeps it at rank 1.

The same insurance works in **both directions**. For *"How much does the EIA expect US marketed
natural gas production to grow in 2026?"* (gold p13), the visual lane ranks a wrong chart page
p25 first (24.21 vs 23.98 — nearly tied); the text lane ranks p13 first decisively (0.811 vs
0.756), and fusion restores p13 to rank 1. Neither lane alone gets both questions right; the
hybrid gets both.

| Retrieved gold page (visual lane won) | Lane-flip case (text lane won) |
|---|---|
| ![Berkshire p5 — earnings table](assets/berkshire_2024_letter_p5.jpg) | ![EIA p13 — natural gas production](assets/eia_short_term_energy_outlook_p13.jpg) |

### Stage 4 — VLM generation (page images → Gemini 2.5 Flash)
Top-5 fused page **images** go to the VLM with a citation-forcing prompt. Actual output:

> "Berkshire's insurance-underwriting earnings in 2024 were $9,020 million. This is an increase
> compared to 2023, when they were $5,428 million **[berkshire_2024_letter p.5]**."

Both figures come from the table on p5 — a text-only pipeline never had them intact. The model
also declines correctly: asked for a number the retrieved pages don't contain, it answered
*"The provided document pages do not contain the forecast WTI spot price for Q2 2026"* instead
of hallucinating one (that answer still scored faithfulness 5/5).

### Stage 5 — Evaluation (10 gold-annotated questions, full corpus)

Retrieval is scored exactly against hand-annotated gold pages (hit@k, MRR); answer quality is
scored by Gemini 2.5 Flash-Lite — deliberately a **different model** from the generator — on
separate 1–5 faithfulness and relevance rubrics, with the gold answer in the judge prompt.

| Metric | Score |
|---|---|
| hit@1 | **0.80** |
| hit@5 | **0.90** |
| MRR | **0.83** |
| Faithfulness (LLM-judge, 1–5) | **5.0** |
| Relevance (LLM-judge, 1–5) | **5.0** |

**Ablation — does the hybrid actually earn its keep?** Same 10 questions, three retrieval
configurations, computed post-hoc from the captured per-lane rankings with zero extra API calls
([`scripts/analyze_eval.py`](../scripts/analyze_eval.py)):

| Retrieval config | hit@1 | hit@5 | MRR |
|---|---|---|---|
| Visual lane only (ColQwen2.5) | 0.70 | 0.90 | 0.78 |
| Text lane only (BGE-small) | 0.60 | 0.90 | 0.75 |
| **Hybrid (RRF fusion)** | **0.80** | **0.90** | **0.83** |

The fused ranking matched or beat the better single lane on **9 of 10** questions, rescuing
2 text-lane misses and 1 visual-lane miss. The two remaining misses are exactly the questions
where *both* lanes rank the wrong page — fusion can't outvote a unanimous mistake.

**Stratified hit@1 — where retrieval degrades:**

| Question type | n | Hybrid | Visual only | Text only |
|---|---|---|---|---|
| Prose / narrative | 6 | **1.00** | 0.83 | 0.83 |
| Numeric table lookup | 4 | **0.50** | 0.50 | 0.25 |

The aggregate 0.80 hides the real story: retrieval is perfect on prose and coin-flip on
dense-table lookups — which is why table-aware indexing leads the next-level solution.

**Answer quality — objective checks beyond the LLM judge:**

| Metric | Score |
|---|---|
| Numeric exact-match (reference key figure present in answer) | **8/8** answered numeric questions |
| Citation precision (cited pages that are gold pages) | **1.00** |
| Gold page cited in the answer | **1.00** |
| Hallucinations when the gold page wasn't retrieved | **0/1** (model refused instead) |
| False refusals on answerable questions | **0/9** |

Financial answers are numbers, so correctness is checked by *parsing*, not by another LLM:
every answered question contains the reference answer's key figure verbatim. The refusal matrix
is the grounding proof — one question's gold page never reached the model, and it said so
instead of reading a plausible number off the nearest chart. Full breakdown:
[`results/metrics_extended.json`](../results/metrics_extended.json).

**Failure analysis** — both hit@1 misses are the same shape: quarterly-statistics-table lookups
(e.g. *"WTI spot price for Q2 2026"*, gold p34 below right). The gold page is a wall of small
numbers with almost no distinguishing text or visual structure; both lanes rank the WTI *price
chart* page p17 (below left) above it. That's the known hard case for page-level retrieval — the
fix would be table-aware indexing (extract tables → index rows/cells), which is the first item
in the next-level solution below.

| What both lanes retrieved (p17, chart) | What the gold answer needed (p34, dense table) |
|---|---|
| ![EIA p17 — WTI chart](assets/eia_short_term_energy_outlook_p17.jpg) | ![EIA p34 — quarterly table](assets/eia_short_term_energy_outlook_p34.jpg) |

Per-question raw results: [`results/eval_results.json`](../results/eval_results.json).

### Stage 6 — Serving
FastAPI REST API + Streamlit chat UI (shows the retrieved page images alongside the answer) +
Docker Compose. Public demo: [Hugging Face Space](https://huggingface.co/spaces/usamahassan965/findociq-demo).
Answers **stream token-by-token**, and every response carries per-stage latency (retrieval ms ·
time-to-first-token · full answer); the API returns the same timings in JSON and exposes a
`/query/stream` endpoint.

## 4. Problems I faced

Each problem is stated plainly, followed by how it was solved — with the real numbers behind
both.

### 1. The answer is one cell in a 40-row table — and every retriever walks past it

**Problem** — Ask *"What is the forecast WTI spot price for Q2 2026?"* and the system fetches
the wrong page. The correct answer is one number inside a huge quarterly statistics table (p34),
but both retrievers — visual and text — rank a WTI price *chart* (p17) above it. The reason is
simple once you see it: the system creates one embedding per *page*, so a table with 40 rows of
numbers gets blended into a single average in which no individual cell stands out. A chart page
about the same topic looks far more "relevant" to the query words. And because *both* lanes make
the same mistake, fusion can't outvote it.

**Solution** — First, measure it honestly instead of hiding it: the stratified eval shows
retrieval is perfect (1.00 hit@1) on prose questions and coin-flip (0.50) on dense-table
lookups — the failure is precisely bounded, not vague. The real fix is a granularity change,
not a better embedding: extract tables during ingestion and index individual rows/cells
alongside whole pages, then route "what was X in Q3"-style numeric questions to that
fine-grained index. That's the top item in the next-level solution below. Meanwhile the system
fails *safely* — see problem 5.

### 2. Two retrievers, two incomparable scoring universes

**Problem** — The two search lanes speak different languages. The visual lane scores pages with
MAX_SIM — an open-ended sum that landed anywhere between 20 and 30 depending on how long the
question is. The text lane returns cosine similarity, always between 0 and 1. You can't average
a "29.47" with a "0.77" — and normalizing doesn't save you: squeeze each lane's 15 candidates
into 0–1 and one unusually high top score crushes everyone else toward zero, so the same page
gets a completely different normalized score from one question to the next.

**Solution** — Stop comparing scores and compare *positions* instead. Reciprocal Rank Fusion
only asks each lane "what's your 1st, 2nd, 3rd choice?" and rewards pages that rank high in
both lists (`score = Σ 1/(60 + rank)`). No scales to reconcile, no training data needed. The
eval proves it works in both directions: on the insurance question the text lane picks the
wrong page (0.778 vs 0.767 — nearly tied) but the visual lane's confident 29.47 keeps the right
page at #1; on the natural-gas question the roles flip — the visual lane errs (24.21 vs 23.98)
and the text lane's clear 0.811 pulls the right page back to the top. Neither lane alone gets
both right; the fusion gets both.

### 3. One NaN embedding took down the entire index

**Problem** — The indexing run crashed because exactly 1 page out of 75 produced embeddings
that were pure `NaN` (not-a-number), which the vector database rejects outright. The root cause
was the number format forced by the old GPU: fp16 can only represent numbers up to 65,504,
while the bf16 format the model was designed for reaches ~10³⁸. Somewhere inside the model one
intermediate value exceeded that ceiling and became "infinity", and a later
`infinity − infinity` step produced NaN — which then contaminated every downstream number for
that page.

**Solution** — Count the NaNs per page to isolate the culprit — it turned out to be a nearly
blank cover page with almost nothing on it. Then replace the broken values with zeros
(`nan_to_num`). That's not a hack, it's provably safe: retrieval scores pages by dot products
against the query, and a zero vector scores exactly 0 against anything — so a blank page can
never outrank a real one. The indexing run completes, and the one affected page simply never
wins retrieval, which is the correct behavior for a blank page.

### 4. The index was born on a GPU the serving machine will never have

**Problem** — Turning a page into embeddings means pushing it through a 3-billion-parameter
vision model — that needs a GPU and 7.5 GB of weights. But the demo runs on a free CPU-only
container that could never load the model, let alone run it. How does a machine that can't run
the embedder serve an index built by it?

**Solution** — Notice the asymmetry: a *page* costs 755 patch vectors through the full model,
but a *question* is only ~30 token vectors — thousands of times cheaper. So the expensive
page-side work happens once, offline, on a borrowed Kaggle GPU, and only the results travel: a
~19 MB artifact (`embeddings.npz` + manifest) instead of a 7.5 GB model. The indexer treats
that artifact as a contract — every page gets a deterministic ID (`uuid5(doc_id,
page_number)`), so re-running ingestion overwrites cleanly instead of duplicating, and the
embeddings can be regenerated on any GPU without touching the serving stack. It's the classic
train-time/serve-time split, applied to retrieval.

### 5. Getting a vision model to say "I don't know"

**Problem** — Show a vision model five financial pages and ask for a number that isn't on them,
and it will happily read a *similar* number off the nearest chart. In finance that's the most
dangerous failure possible: the answer looks plausible, cites a real page, and is wrong.

**Solution** — Two layers. First, the generation prompt demands a `[doc p.N]` citation for
every claim and gives the model an explicit exit: if the pages don't contain the answer, say
so. Second, the UI shows the actual retrieved page images next to every answer, so any number
is visually checkable in seconds. It measurably works: when retrieval surfaced the WTI chart
page instead of the statistics table (problem 1), the model answered *"The provided document
pages do not contain the forecast WTI spot price for Q2 2026"* — a correct refusal instead of a
confident wrong number. Across the whole eval: 0 hallucinations, 0 false refusals.

### 6. An evaluation you can trust on 20 API calls a day

**Problem** — Two traps hide in "just have an LLM grade the answers." Trap one: if the same
model both writes and grades the answers, it grades its own homework — scores come out
inflated. Trap two: the free API tier allows roughly 20 requests per day per model, and one
full eval needs exactly 20 calls (10 answers + 10 judgments). A single crash mid-run wastes
calls you cannot get back until tomorrow.

**Solution** — Separate the roles: Gemini 2.5 Flash writes the answers, Flash-Lite — a
different model — judges them, with the gold answer in hand and two separate 1–5 rubrics
(faithfulness and relevance) instead of one vague "quality" score. Then make the loop
unbreakable: every completed answer and judgment is saved to disk immediately, reruns skip
anything already finished, and rate-limit errors trigger wait-and-retry instead of a crash. A
run that dies at question 9 resumes at question 9 — zero wasted quota.

## 5. Next-level solution

What separates this prototype from a production document-intelligence system — each item targets
a measured limitation, not a hypothetical one.

- **Table-aware indexing.** Extract tables at parse time and index row/cell-level granules
  alongside page-level vectors, then route numeric "what was X in Q3" queries to the
  fine-grained index. Directly attacks the only failure mode the eval found (both misses were
  dense-table lookups losing to chart pages).
- **Reranking stage after fusion.** RRF discards score magnitude by design; a cross-encoder
  reranker over the top-15 fused candidates re-reads the actual page content against the query
  and would likely recover the p34-vs-p17 confusions without touching the indexes.
- **Multivector compression for scale.** 755 × 128 fp16 vectors ≈ 189 KB per page — fine at
  75 pages, untenable at 100K. Binary quantization plus two-stage retrieval (mean-pooled
  single-vector recall, then exact MAX_SIM rescoring on the shortlist) cuts storage ~32× with
  minimal ranking loss.
- **Eval at scale, stratified by page type.** 10 hand-written questions prove the pipeline; they
  can't measure it. Generate synthetic QA pairs stratified across prose, table, and chart pages
  (with human spot-checks), so per-page-type hit rates expose exactly where retrieval degrades.
- **Query-adaptive lane weighting.** Today both lanes vote equally in RRF. A lightweight query
  router (is this visual, numeric, or textual intent?) can weight the lanes per query — e.g.
  trust the text lane more on verbatim-phrase questions, the visual lane on chart questions.
- **Production hardening.** Token-by-token streaming and per-stage latency instrumentation are
  now shipped (UI + API); still ahead: retrieved-page image caching, request tracing, and
  retrieval-quality dashboards (rank distributions, per-lane agreement rate) so drift is
  visible before users report it.
