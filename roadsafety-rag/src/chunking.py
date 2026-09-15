"""
Stage 2: Chunking + metadata tagging
Split ingested documents into small chunks (road rules are short/discrete,
so start small) and tag each chunk with driver_stage + topic metadata.

This is the file to experiment with: try CHUNK_SIZE = 100 / 200 / 300 / 500
and see which keeps individual rules intact without splitting mid-rule.
"""
import json
from pathlib import Path
from ingestion import load_documents

CHUNK_SIZE = 200      # target words per chunk
CHUNK_OVERLAP = 40    # words of overlap between consecutive chunks

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- placeholder keyword-based tagger -----------------------------------
# Replace this with something smarter once you have real data:
# a small zero-shot classifier, or a manual tagging pass.
STAGE_KEYWORDS = {
    "learner": ["learner", "l plate", "l-plate", "supervising driver"],
    "P1": ["p1", "red p", "probationary (first year)"],
    "P2": ["p2", "green p"],
    "full": ["full licence", "full license", "open licence"],
}

TOPIC_KEYWORDS = {
    "speed": ["speed limit", "km/h", "speeding"],
    "phone_use": ["mobile phone", "hands-free", "phone use"],
    "parking": ["parking", "park", "no standing"],
    "signage": ["sign", "signal", "give way", "stop sign"],
    "alcohol_drugs": ["bac", "blood alcohol", "drug driving", "zero bac"],
    "seatbelts": ["seatbelt", "seat belt", "restraint"],
}


def tag_stage(text: str) -> str:
    lowered = text.lower()
    for stage, keywords in STAGE_KEYWORDS.items():
        if any(k in lowered for k in keywords):
            return stage
    return "all"  # applies to all driver stages / not stage-specific


def tag_topic(text: str) -> str:
    lowered = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        if any(k in lowered for k in keywords):
            return topic
    return "general"


# --- chunking -------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start = end - overlap  # step forward, keeping overlap
    return chunks


def build_chunks(docs: list[dict]) -> list[dict]:
    """Return list of {chunk_id, doc_id, source, text, driver_stage, topic,
    url, date_accessed} - provenance carried from ingestion, stage/topic
    tagged per-chunk (falls back to the doc-level tag from sources.csv if
    the keyword tagger finds nothing more specific)."""
    all_chunks = []
    for doc in docs:
        pieces = chunk_text(doc["text"])
        for i, piece in enumerate(pieces):
            stage = tag_stage(piece)
            topic = tag_topic(piece)
            all_chunks.append({
                "chunk_id": f"{doc['doc_id']}_{i}",
                "doc_id": doc["doc_id"],
                "source": doc["source"],
                "text": piece,
                "driver_stage": stage if stage != "all" else (doc.get("doc_driver_stage") or "all"),
                "topic": topic if topic != "general" else (doc.get("doc_topic") or "general"),
                "url": doc.get("url", ""),
                "date_accessed": doc.get("date_accessed", ""),
            })
    return all_chunks


def save_chunks(chunks: list[dict], path: Path = PROCESSED_DIR / "chunked_docs.jsonl"):
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print(f"Saved {len(chunks)} chunks to {path}")


if __name__ == "__main__":
    docs = load_documents()
    chunks = build_chunks(docs)
    save_chunks(chunks)
