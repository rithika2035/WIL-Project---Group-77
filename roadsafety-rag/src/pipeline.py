"""
End-to-end CLI.

Usage:
    python pipeline.py build              # raw PDFs: ingest -> chunk -> embed -> index
    python pipeline.py build-collection   # pre-chunked data/collection.csv -> embed -> index
    python pipeline.py ask "question"     # retrieve -> generate -> print answer
"""
import sys

from ingestion import load_documents
from chunking import build_chunks, save_chunks
from embed_store import build_index
from retrieval import retrieve
from generation import generate_answer
import load_collection


def build():
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    chunks = build_chunks(docs)
    save_chunks(chunks)
    build_index(chunks)


def build_from_collection():
    chunks = load_collection.load_chunks()
    print(f"Loaded {len(chunks)} pre-chunked passages from data/collection.csv")
    build_index(chunks)


def ask(question: str):
    hits = retrieve(question)
    result = generate_answer(question, hits)

    print("\n--- Retrieved chunks ---")
    for h in hits:
        print(f"[{h['distance']:.3f}] {h['metadata']} -> {h['text'][:80]}...")

    print("\n--- Answer ---")
    print(result["answer"])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command == "build":
        build()
    elif command == "build-collection":
        build_from_collection()
    elif command == "ask":
        if len(sys.argv) < 3:
            print("Usage: python pipeline.py ask \"your question\"")
            sys.exit(1)
        ask(sys.argv[2])
    else:
        print(__doc__)
