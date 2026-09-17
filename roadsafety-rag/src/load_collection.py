"""
Alternative to ingestion.py + chunking.py, for when your data already comes
as a pre-chunked, pre-tagged passage collection (data/collection.csv) rather
than raw PDFs you need to chunk yourself. This is the same shape as Walert's
own passage collection: fixed passages with a stable id and metadata columns,
built by hand or by a separate offline chunking pass.

Expected CSV columns (from your collection.csv):
    passage_id, passage_text, source_doc, chapter, section_title,
    page_number, jurisdiction, driver_stage, topic, content_type, word_count

Note: driver_stage can hold MULTIPLE stages separated by semicolons, e.g.
"learner;P1;P2" - a passage that applies to several stages at once. This is
handled as a raw string here and matched with substring/membership logic in
retrieval.py, not exact equality, since Chroma metadata values must be
scalars (no lists).
"""
import csv
from pathlib import Path

COLLECTION_CSV = Path(__file__).resolve().parent.parent / "data" / "collection.csv"


def load_chunks(path: Path = COLLECTION_CSV) -> list[dict]:
    """Return list of chunk dicts in the shape embed_store.build_index() expects."""
    chunks = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            chunks.append({
                "chunk_id": row["passage_id"],
                "doc_id": row["source_doc"],
                "source": row["source_doc"],
                "text": row["passage_text"],
                "driver_stage": row.get("driver_stage", "all") or "all",
                "topic": row.get("topic", "general") or "general",
                "url": "",           # not in this CSV - log the source doc in data/sources.csv instead
                "date_accessed": "",
                "chapter": row.get("chapter", ""),
                "section_title": row.get("section_title", ""),
                "page_number": row.get("page_number", ""),
                "jurisdiction": row.get("jurisdiction", ""),
                "content_type": row.get("content_type", ""),
            })
    return chunks


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} passages from {COLLECTION_CSV}")
    print("Example:", chunks[0])
