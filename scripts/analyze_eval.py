"""Post-hoc extended metrics over the saved eval artifacts.

Computes, with zero API calls:
  1. Ablation study      — visual-only / text-only / hybrid retrieval (hit@1, hit@5, MRR)
  2. Fusion gain         — per-question: does RRF match/beat the best single lane?
  3. Stratified hit@1    — numeric table-lookup questions vs prose/narrative questions
  4. Numeric exact-match — does the answer contain the reference answer's key figure(s)?
  5. Citation metrics    — precision/recall of [doc p.N] citations vs gold pages
  6. Refusal matrix      — answered/refused x context-answerable/unanswerable

Inputs : results/eval_results.json, results/lane_rankings.json, data/eval/eval_questions.json
Output : results/metrics_extended.json (+ human-readable report on stdout)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Question strata: a question is a "table lookup" when its canonical source is a
# dense statistics table (the answer is a specific figure inside a table), else "prose".
# Index-aligned with data/eval/eval_questions.json.
QUESTION_STRATA = [
    "table",   # Q1  insurance-underwriting earnings   (Berkshire p5 earnings table)
    "prose",   # Q2  operating businesses decline      (narrative, p4)
    "prose",   # Q3  only cash dividend                (narrative, p6)
    "table",   # Q4  1965 vs S&P 500                   (p14 annual performance table)
    "table",   # Q5  Brent 2026 forecast               (p3 summary / p34 stats table)
    "prose",   # Q6  oil market volatility cause       (narrative, p6)
    "prose",   # Q7  gas production growth 2026        (narrative + chart, p13)
    "prose",   # Q8  Henry Hub May average             (narrative, p12)
    "table",   # Q9  WTI Q2 2026 forecast              (p34 quarterly stats table)
    "prose",   # Q10 coal consumption decline          (narrative + chart, p15)
]

REFUSAL_MARKERS = (
    "do not contain",
    "does not contain",
    "not provided in the",
    "cannot be determined from",
)

CITATION_RE = re.compile(r"\[([a-z0-9_]+)\s+p\.(\d+)\]", re.IGNORECASE)
NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def extract_numbers(text: str) -> list[str]:
    """Numeric tokens normalised (commas stripped, trailing dot trimmed)."""
    out = []
    for tok in NUMBER_RE.findall(text):
        tok = tok.replace(",", "").rstrip(".")
        if tok:
            out.append(tok)
    return out


def key_figures(reference: str) -> list[str]:
    """Reference-answer figures excluding bare years (which appear in the question)."""
    return [n for n in extract_numbers(reference) if not re.fullmatch(r"(19|20)\d{2}", n)]


def rank_of_gold(ranking: list[str], gold: list[str]) -> int | None:
    """1-based rank of the first gold page in a ranking, or None."""
    for i, page in enumerate(ranking, start=1):
        if page in gold:
            return i
    return None


def retrieval_metrics(rankings: list[list[str]], golds: list[list[str]]) -> dict:
    hits1 = hits5 = 0
    rr_sum = 0.0
    for ranking, gold in zip(rankings, golds):
        rank = rank_of_gold(ranking[:5], gold)
        if rank == 1:
            hits1 += 1
        if rank is not None:
            hits5 += 1
            rr_sum += 1.0 / rank
    n = len(rankings)
    return {"hit@1": hits1 / n, "hit@5": hits5 / n, "mrr": round(rr_sum / n, 4)}


def main() -> None:
    eval_results = json.loads((ROOT / "results" / "eval_results.json").read_text(encoding="utf-8"))
    lanes = json.loads((ROOT / "results" / "lane_rankings.json").read_text(encoding="utf-8"))
    questions = json.loads((ROOT / "data" / "eval" / "eval_questions.json").read_text(encoding="utf-8"))

    results = eval_results["results"]
    lane_qs = lanes["questions"]
    assert len(results) == len(lane_qs) == len(questions) == len(QUESTION_STRATA)

    golds = [q["gold_pages"] for q in questions]

    # ---- 1. Ablation ----------------------------------------------------------------
    visual = [[e["page_id"] for e in q["visual_top5"]] for q in lane_qs]
    text = [[e["page_id"] for e in q["text_top5"]] for q in lane_qs]
    fused = [[e["page_id"] for e in q["fused_top5"]] for q in lane_qs]
    ablation = {
        "visual_only": retrieval_metrics(visual, golds),
        "text_only": retrieval_metrics(text, golds),
        "hybrid_rrf": retrieval_metrics(fused, golds),
    }

    # ---- 2. Fusion gain --------------------------------------------------------------
    fusion = {"matches_best_lane": 0, "beats_worst_lane": 0, "rescued_visual_miss": 0,
              "rescued_text_miss": 0, "unrecoverable_both_lanes_wrong": 0}
    for v, t, f, gold in zip(visual, text, fused, golds):
        rv, rt, rf = rank_of_gold(v, gold), rank_of_gold(t, gold), rank_of_gold(f, gold)
        rr = lambda r: 0.0 if r is None else 1.0 / r
        if rr(rf) >= max(rr(rv), rr(rt)):
            fusion["matches_best_lane"] += 1
        if rr(rf) > min(rr(rv), rr(rt)):
            fusion["beats_worst_lane"] += 1
        if rv != 1 and rf == 1:
            fusion["rescued_visual_miss"] += 1
        if rt != 1 and rf == 1:
            fusion["rescued_text_miss"] += 1
        if rv != 1 and rt != 1 and rf != 1:
            fusion["unrecoverable_both_lanes_wrong"] += 1

    # ---- 3. Stratified hit@1 (hybrid, plus per lane) ---------------------------------
    strata: dict[str, dict] = {}
    for stratum in ("table", "prose"):
        idx = [i for i, s in enumerate(QUESTION_STRATA) if s == stratum]
        strata[stratum] = {
            "n": len(idx),
            "hybrid": retrieval_metrics([fused[i] for i in idx], [golds[i] for i in idx]),
            "visual_only": retrieval_metrics([visual[i] for i in idx], [golds[i] for i in idx]),
            "text_only": retrieval_metrics([text[i] for i in idx], [golds[i] for i in idx]),
        }

    # ---- 4-6. Answer-quality metrics -------------------------------------------------
    numeric_checked = numeric_matched = 0
    cit_prec_sum = cit_rec_hits = cit_answered = 0
    matrix = {"answered_when_answerable": 0, "refused_when_answerable": 0,
              "refused_when_unanswerable": 0, "hallucinated_when_unanswerable": 0}
    per_question = []

    for i, (res, q) in enumerate(zip(results, questions)):
        answer, gold = res["answer"], q["gold_pages"]
        refused = any(m in answer.lower() for m in REFUSAL_MARKERS)
        context_answerable = any(g in res["retrieved_pages"] for g in gold)

        if context_answerable:
            matrix["refused_when_answerable" if refused else "answered_when_answerable"] += 1
        else:
            matrix["refused_when_unanswerable" if refused else "hallucinated_when_unanswerable"] += 1

        detail = {"q": i + 1, "stratum": QUESTION_STRATA[i], "refused": refused,
                  "context_answerable": context_answerable}

        if not refused:
            # numeric exact-match on the reference's key figures
            figures = key_figures(q["reference_answer"])
            if figures:
                answer_nums = set(extract_numbers(answer))
                primary_ok = figures[0] in answer_nums
                numeric_checked += 1
                numeric_matched += primary_ok
                detail["key_figure"] = figures[0]
                detail["numeric_match"] = primary_ok

            # citation precision / recall
            cited = {f"{doc}::p{page}" for doc, page in CITATION_RE.findall(answer)}
            if cited:
                cit_answered += 1
                prec = len(cited & set(gold)) / len(cited)
                cit_prec_sum += prec
                cit_rec_hits += bool(cited & set(gold))
                detail["cited"] = sorted(cited)
                detail["citation_precision"] = prec

        per_question.append(detail)

    out = {
        "ablation": ablation,
        "fusion_gain": fusion,
        "stratified_hit_rates": strata,
        "numeric_exact_match": {
            "n_checked": numeric_checked,
            "matched": numeric_matched,
            "rate": round(numeric_matched / numeric_checked, 4) if numeric_checked else None,
        },
        "citations": {
            "n_answered_with_citations": cit_answered,
            "precision_avg": round(cit_prec_sum / cit_answered, 4) if cit_answered else None,
            "gold_cited_rate": round(cit_rec_hits / cit_answered, 4) if cit_answered else None,
        },
        "refusal_matrix": matrix,
        "per_question": per_question,
    }

    out_path = ROOT / "results" / "metrics_extended.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("=== Ablation (10 questions) ===")
    for name, m in ablation.items():
        print(f"  {name:<12} hit@1 {m['hit@1']:.2f}  hit@5 {m['hit@5']:.2f}  MRR {m['mrr']:.2f}")
    print("=== Fusion gain ===")
    for k, v in fusion.items():
        print(f"  {k}: {v}")
    print("=== Stratified hit@1 (hybrid | visual | text) ===")
    for s, d in strata.items():
        print(f"  {s:<6} (n={d['n']})  {d['hybrid']['hit@1']:.2f} | "
              f"{d['visual_only']['hit@1']:.2f} | {d['text_only']['hit@1']:.2f}")
    print("=== Answer quality ===")
    print(f"  numeric exact-match : {numeric_matched}/{numeric_checked}")
    print(f"  citation precision  : {out['citations']['precision_avg']}")
    print(f"  gold page cited     : {out['citations']['gold_cited_rate']}")
    print(f"  refusal matrix      : {matrix}")
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
