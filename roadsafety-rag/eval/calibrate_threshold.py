"""
Calibrate generation.DISTANCE_THRESHOLD against real labelled data instead
of guessing. Runs retrieval (not generation) for every question in
eval/test_set_from_topics.csv, records the top-1 distance, then sweeps
candidate thresholds and reports which one best separates "should answer"
from "should abstain" questions.

Run this BEFORE trusting DISTANCE_THRESHOLD in generation.py. Re-run it
whenever you change the embedding model, chunk size, or add more data to
collection.csv - the right threshold shifts with all of these.

Usage:
    python eval/convert_topics_to_test_set.py   # if you haven't already
    python eval/calibrate_threshold.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retrieval import retrieve


def collect_top1_distances(test_set_path: str) -> pd.DataFrame:
    from embed_store import get_collection

    collection = get_collection()
    count = collection.count()
    if count == 0:
        print(
            "\nERROR: the Chroma collection has 0 items in it.\n"
            "This is why every distance came back as infinity - there was\n"
            "nothing to retrieve against. Run this first, in this same\n"
            "terminal/venv, then re-run calibrate_threshold.py:\n\n"
            "    python src/pipeline.py build-collection\n\n"
            "If you already ran that in a different session, note that\n"
            "data/chroma_db/ is gitignored (it's a local index, not meant\n"
            "to be committed) - if you re-cloned the repo, switched\n"
            "machines, or deleted the data/ folder, you'll need to rebuild\n"
            "the index locally every time.\n"
        )
        sys.exit(1)
    print(f"Collection has {count} items - proceeding.\n")

    df = pd.read_csv(test_set_path)
    records = []

    for _, row in df.iterrows():
        hits = retrieve(row["question"], top_k=3, filter_stage=False)
        top_distance = hits[0]["distance"] if hits else float("inf")
        records.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "expect_answerable": row["expect_answerable"],
            "top1_distance": top_distance,
        })

    return pd.DataFrame(records)


def sweep_thresholds(df: pd.DataFrame, candidates=None):
    finite = df[df["top1_distance"].apply(lambda x: x not in (float("inf"), float("-inf")) and pd.notna(x))]
    n_dropped = len(df) - len(finite)
    if n_dropped:
        print(f"WARNING: {n_dropped} question(s) returned zero retrieval hits "
              f"(infinite distance) and were excluded from the threshold sweep. "
              f"Check eval/calibration_distances.csv for which ones.")
    if finite.empty:
        print("ERROR: no question produced a finite distance - nothing to calibrate against.")
        sys.exit(1)

    if candidates is None:
        # sweep a range based on whatever distances actually showed up
        lo, hi = finite["top1_distance"].min(), finite["top1_distance"].max()
        step = max((hi - lo) / 20, 0.01)
        candidates = [round(lo + i * step, 3) for i in range(21)]

    results = []
    for t in candidates:
        predicted_answerable = finite["top1_distance"] <= t
        actual_answerable = finite["expect_answerable"] == "yes"
        accuracy = (predicted_answerable == actual_answerable).mean()
        # also report false-answer rate (predicted yes when should abstain) -
        # the riskier failure mode, worth weighting more than false-abstain
        false_answer_rate = ((predicted_answerable) & (~actual_answerable)).mean()
        results.append({
            "threshold": t,
            "accuracy": accuracy,
            "false_answer_rate": false_answer_rate,
        })

    return pd.DataFrame(results).sort_values("accuracy", ascending=False)


if __name__ == "__main__":
    test_set_path = sys.argv[1] if len(sys.argv) > 1 else "eval/test_set_from_topics.csv"

    print("Retrieving top-1 distance for every question (this calls the "
          "embedding model, no LLM/Ollama needed for this step)...")
    distances_df = collect_top1_distances(test_set_path)
    distances_df.to_csv("eval/calibration_distances.csv", index=False)
    print(f"Wrote per-question distances to eval/calibration_distances.csv")

    print("\n=== Distance summary by expected answerability ===")
    print(distances_df.groupby("expect_answerable")["top1_distance"].describe())

    print("\n=== Threshold sweep (best accuracy first) ===")
    sweep_df = sweep_thresholds(distances_df)
    print(sweep_df.head(10).to_string(index=False))

    best = sweep_df.iloc[0]
    print(f"\nRecommended DISTANCE_THRESHOLD: {best['threshold']} "
          f"(accuracy {best['accuracy']:.2%}, false-answer rate {best['false_answer_rate']:.2%})")
    print("Update DISTANCE_THRESHOLD in src/generation.py with this value.")
