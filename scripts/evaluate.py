"""Run the evaluation pipeline.

Usage: python scripts/evaluate.py data/eval/eval_questions.json
"""

import argparse
import json
from pathlib import Path

from findociq.evaluation.evaluate import run_evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Path to eval questions JSON")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-judge scoring")
    parser.add_argument("--out", type=Path, default=Path("results/eval_results.json"))
    args = parser.parse_args()

    summary = run_evaluation(args.dataset, judge=not args.no_judge, out_path=args.out)
    print(json.dumps(summary, indent=2))
    print(f"\nFull per-question results written to {args.out}")


if __name__ == "__main__":
    main()
