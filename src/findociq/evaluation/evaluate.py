"""Stage 5 - Evaluation: retrieval metrics + LLM-judge answer scoring.

Retrieval: hit@1, hit@k, MRR against annotated gold pages.
Generation: an LLM judge scores each answer 1-5 for faithfulness (grounded
in the retrieved pages?) and relevance (does it answer the question?).
The judge runs on the same free provider as generation, so evaluation
costs nothing.

Eval dataset format (data/eval/eval_questions.json):
[
  {
    "question": "What was total revenue in FY2024?",
    "gold_pages": ["annual_report_2024::p12"],
    "reference_answer": "optional"
  }
]
"""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from findociq.generation.vlm_client import VLMClient, get_vlm_client
from findociq.retrieval.hybrid_retriever import HybridRetriever, RetrievedPage

JUDGE_PROMPT = """You are grading a RAG system's answer. Score two dimensions from 1 to 5:

faithfulness: Is every claim in the answer supported by the provided page images? (5 = fully grounded, 1 = hallucinated)
relevance: Does the answer actually address the question? (5 = fully, 1 = not at all)

Question: {question}

Answer to grade:
{answer}

Respond with exactly two lines:
faithfulness: <score>
relevance: <score>"""


@dataclass
class EvalResult:
    question: str
    retrieved_pages: list[str]
    gold_pages: list[str]
    hit_at_1: bool
    hit_at_k: bool
    reciprocal_rank: float
    answer: str
    faithfulness: float | None
    relevance: float | None


def _retrieval_metrics(retrieved: list[str], gold: list[str]) -> tuple[bool, bool, float]:
    gold_set = set(gold)
    hit_at_1 = bool(retrieved) and retrieved[0] in gold_set
    hit_at_k = any(pid in gold_set for pid in retrieved)
    rr = 0.0
    for rank, pid in enumerate(retrieved, start=1):
        if pid in gold_set:
            rr = 1.0 / rank
            break
    return hit_at_1, hit_at_k, rr


def _judge(client: VLMClient, question: str, answer: str, pages: list[RetrievedPage]) -> tuple[float | None, float | None]:
    raw = client.answer(JUDGE_PROMPT.format(question=question, answer=answer), pages)
    faith = re.search(r"faithfulness:\s*([1-5])", raw, re.IGNORECASE)
    rel = re.search(r"relevance:\s*([1-5])", raw, re.IGNORECASE)
    return (
        float(faith.group(1)) if faith else None,
        float(rel.group(1)) if rel else None,
    )


def run_evaluation(eval_path: Path, judge: bool = True, out_path: Path | None = None) -> dict:
    dataset = json.loads(eval_path.read_text(encoding="utf-8"))
    retriever = HybridRetriever()
    client = get_vlm_client()

    results: list[EvalResult] = []
    for item in tqdm(dataset, desc="evaluating"):
        pages = retriever.retrieve(item["question"])
        retrieved_ids = [p.page_id for p in pages]
        hit1, hitk, rr = _retrieval_metrics(retrieved_ids, item["gold_pages"])

        answer = client.answer(item["question"], pages)
        faith, rel = _judge(client, item["question"], answer, pages) if judge else (None, None)

        results.append(
            EvalResult(
                question=item["question"],
                retrieved_pages=retrieved_ids,
                gold_pages=item["gold_pages"],
                hit_at_1=hit1,
                hit_at_k=hitk,
                reciprocal_rank=rr,
                answer=answer,
                faithfulness=faith,
                relevance=rel,
            )
        )

    n = len(results)
    judged = [r for r in results if r.faithfulness is not None]
    summary = {
        "n_questions": n,
        "hit@1": sum(r.hit_at_1 for r in results) / n,
        "hit@k": sum(r.hit_at_k for r in results) / n,
        "mrr": sum(r.reciprocal_rank for r in results) / n,
        "faithfulness_avg": (
            sum(r.faithfulness for r in judged) / len(judged) if judged else None
        ),
        "relevance_avg": (
            sum(r.relevance for r in judged if r.relevance) / len(judged) if judged else None
        ),
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"summary": summary, "results": [asdict(r) for r in results]}, indent=2),
            encoding="utf-8",
        )
    return summary
