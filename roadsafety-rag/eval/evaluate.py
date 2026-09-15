"""
Simple top-k retrieval accuracy check (doc-level) against eval/test_set_template.csv.
For the full Phase 3 metric suite (effectiveness, faithfulness, stage/category
precision, source attribution), use evaluate_full.py instead - this script is
just the quick retrieval sanity check.

Usage:
    python eval/evaluate.py [path/to/test_set.csv] [top_k]
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retrieval import retrieve


def evaluate(test_set_path: str, top_k: int = 3):
    df = pd.read_csv(test_set_path)
    df = df[df["expect_answerable"] == "yes"]  # only score retrieval on answerable Qs
    correct = 0

    for _, row in df.iterrows():
        hits = retrieve(row["question"], top_k=top_k, filter_stage=False)
        retrieved_docs = [h["metadata"]["doc_id"] for h in hits]
        hit = row["gold_source_doc_id"] in retrieved_docs
        correct += int(hit)
        print(f"{'✓' if hit else '✗'} {row['question_id']}: {row['question']}")

    accuracy = correct / len(df) if len(df) else 0
    print(f"\nTop-{top_k} doc-level retrieval accuracy: {accuracy:.2%} ({correct}/{len(df)})")


if __name__ == "__main__":
    test_set_path = sys.argv[1] if len(sys.argv) > 1 else "eval/test_set_template.csv"
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    evaluate(test_set_path, top_k)
