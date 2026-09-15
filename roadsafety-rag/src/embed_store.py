"""
Stage 3: Embedding + vector store
Embed chunks with sentence-transformers and upsert into a persistent
Chroma collection, storing driver_stage/topic as metadata for filtering.
"""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "road_rules"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL_NAME)
    return _model


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(COLLECTION_NAME)


def load_chunks(path: Path = PROCESSED_DIR / "chunked_docs.jsonl") -> list[dict]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_index(chunks: list[dict] | None = None):
    if chunks is None:
        chunks = load_chunks()

    if not chunks:
        print("No chunks to index. Run chunking.py first.")
        return

    model = get_model()
    collection = get_collection()

    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    collection.upsert(
        ids=[c["chunk_id"] for c in chunks],
        embeddings=embeddings,
        documents=texts,
        metadatas=[{
            "doc_id": c["doc_id"],
            "source": c["source"],
            "driver_stage": c["driver_stage"],
            "topic": c["topic"],
            "url": c.get("url", ""),
            "date_accessed": c.get("date_accessed", ""),
        } for c in chunks],
    )
    print(f"Indexed {len(chunks)} chunks into Chroma collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    build_index()
