"""
Phase 3 - full evaluation suite.
Runs every question in the test set through the whole pipeline (retrieve +
generate) and reports the metrics the checklist asks for:

  - Effectiveness: answered / correctly abstained / wrongly abstained / wrongly answered
  - Stage/category precision: does the top retrieved chunk match the expected
    driver_stage and topic (this is the project's core novel metric)
  - Source attribution correctness: does the top retrieved chunk come from
    the expected gold document
  - Faithfulness: NOT fully automatable with this stack. This script flags
    each answered question for manual review (does the answer's claim match
    what the cited source actually says) and writes a `faithful` column you
    fill in by hand - see README note below.

Re-run this after every pipeline change (chunk size, prompt tweak, filtering
logic, etc.) and diff the summary numbers to justify design decisions in the
report ("smaller chunks improved stage precision by X%").

Usage:
    python eval/evaluate_full.py [path/to/test_set.csv]
Outputs:
    eval/results_<timestamp>.csv  (per-question breakdown, incl. blank
                                    'faithful' column to fill in manually)
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retrieval import retrieve
from generation import generate_answer


def classify_effectiveness(expect_answerable: str, used_fallback: bool) -> str:
    expected_yes = expect_answerable.strip().lower() == "yes"
    if expected_yes and not used_fallback:
        return "correctly_answered"
    if expected_yes and used_fallback:
        return "wrongly_abstained"
    if not expected_yes and used_fallback:
        return "correctly_abstained"
    return "wrongly_answered"  # expected abstain but the system answered - riskiest failure mode


def stage_sets_overlap(stage_a: str, stage_b: str) -> bool:
    """Both sides can be semicolon-separated (e.g. 'learner;P1' vs 'P1;P2').
    Match if either is tagged 'all' or the sets share any stage."""
    set_a = {s.strip() for s in str(stage_a).split(";") if s.strip()}
    set_b = {s.strip() for s in str(stage_b).split(";") if s.strip()}
    if "all" in set_a or "all" in set_b:
        return True
    return bool(set_a & set_b)


def run_eval(test_set_path: str) -> pd.DataFrame:
    df = pd.read_csv(test_set_path)
    rows = []

    for _, row in df.iterrows():
        hits = retrieve(row["question"], top_k=3, filter_stage=True)
        result = generate_answer(row["question"], hits)
        top_hit = hits[0] if hits else None

        effectiveness = classify_effectiveness(row["expect_answerable"], result["used_fallback"])

        stage_match = (
            top_hit is not None
            and str(row.get("expected_driver_stage", "")).strip()
            and stage_sets_overlap(top_hit["metadata"]["driver_stage"], row["expected_driver_stage"])
        )
        topic_match = (
            top_hit is not None
            and str(row.get("expected_topic", "")).strip()
            and top_hit["metadata"]["topic"] == row["expected_topic"]
        )
        source_correct = (
            top_hit is not None
            and str(row.get("gold_source_doc_id", "")).strip()
            and top_hit["metadata"]["doc_id"] == row["gold_source_doc_id"]
        )

        rows.append({
            "question_id": row["question_id"],
            "persona": row.get("persona", ""),
            "question_type": row.get("question_type", ""),  # known/inferred/out_of_kb, if present
            "question": row["question"],
            "effectiveness": effectiveness,
            "stage_match": stage_match,
            "topic_match": topic_match,
            "source_attribution_correct": source_correct,
            "answer": result["answer"],
            "top_retrieved_doc": top_hit["metadata"]["doc_id"] if top_hit else None,
            "faithful": "",  # fill in manually: does the answer actually match the cited source?
        })

    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame):
    n = len(results)
    print("\n=== Effectiveness ===")
    print(results["effectiveness"].value_counts().to_string())

    answerable_mask = results["effectiveness"].isin(["correctly_answered", "wrongly_abstained"])
    if answerable_mask.sum():
        print("\n=== Stage/category precision (on questions expected to be answerable) ===")
        print(f"Stage match:  {results.loc[answerable_mask, 'stage_match'].mean():.2%}")
        print(f"Topic match:  {results.loc[answerable_mask, 'topic_match'].mean():.2%}")
        print(f"Source attribution correct: {results.loc[answerable_mask, 'source_attribution_correct'].mean():.2%}")

    if "question_type" in results.columns and results["question_type"].notna().any():
        print("\n=== Effectiveness by question type (known vs inferred vs out_of_kb) ===")
        print("NOTE: 'inferred' questions require combining multiple facts/passages -")
        print("expect these to score lower than 'known' questions with this pipeline.")
        print(results.groupby("question_type")["effectiveness"].value_counts().to_string())

    print(f"\nTotal questions evaluated: {n}")
    print("NOTE: 'faithful' column is blank - fill it in by hand after reading "
          "each answer against its cited source, then recompute a faithfulness % yourself.")


if __name__ == "__main__":
    test_set_path = sys.argv[1] if len(sys.argv) > 1 else "eval/test_set_from_topics.csv"
    results = run_eval(test_set_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path("eval") / f"results_{timestamp}.csv"
    results.to_csv(out_path, index=False)
    print(f"Wrote per-question results to {out_path}")

    print_summary(results)
