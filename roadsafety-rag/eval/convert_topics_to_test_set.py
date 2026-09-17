"""
Convert eval/topics.csv (question_type: known / inferred / out_of_kb) into
the schema eval/evaluate_full.py expects (expect_answerable: yes/no).

Mapping:
    known, inferred  -> expect_answerable = yes  (should be answerable from the KB;
                         "inferred" questions require combining >1 fact/passage,
                         which is harder - see NOTE below)
    out_of_kb         -> expect_answerable = no   (system should abstain)

NOTE on "inferred" questions (e.g. "I'm 19 on my P1, had one beer, can I drive
3 friends home?" - requires combining the P1 zero-BAC rule AND the P1 peer
passenger rule): a simple top-k retrieval + single-pass generation pipeline
often struggles with these because no single passage contains the full
answer. Track known vs inferred accuracy SEPARATELY in your results (this
script keeps question_type as a column so you can group by it) - conflating
them will make your effectiveness numbers look worse than they are on the
easier "known" questions, and better than they are on "inferred" ones.

Usage:
    python eval/convert_topics_to_test_set.py
Outputs:
    eval/test_set_from_topics.csv
"""
import csv
from pathlib import Path

IN_PATH = Path(__file__).resolve().parent / "topics.csv"
OUT_PATH = Path(__file__).resolve().parent / "test_set_from_topics.csv"

ANSWERABLE_MAP = {
    "known": "yes",
    "inferred": "yes",
    "out_of_kb": "no",
}


def convert(in_path: Path = IN_PATH, out_path: Path = OUT_PATH):
    with open(in_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out_rows = []
    for r in rows:
        out_rows.append({
            "question_id": r["question_id"],
            "persona": r["persona"],
            "question": r["question"],
            "expect_answerable": ANSWERABLE_MAP.get(r["question_type"], "yes"),
            "expected_driver_stage": r["expected_driver_stage"],
            "expected_topic": r["expected_topic"],
            "gold_source_doc_id": "",  # topics.csv doesn't carry this - add manually if you want source-attribution scoring
            "question_type": r["question_type"],   # kept for grouping known/inferred/out_of_kb in analysis
            "variant": r["variant"],                # original/paraphrase/focused
        })

    fieldnames = ["question_id", "persona", "question", "expect_answerable",
                  "expected_driver_stage", "expected_topic", "gold_source_doc_id",
                  "question_type", "variant"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} questions to {out_path}")


if __name__ == "__main__":
    convert()
