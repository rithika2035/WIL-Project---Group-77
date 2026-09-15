"""
Stage 4: Retrieval
Embed the incoming question and query Chroma, with optional filtering/
boosting by driver_stage when the question/persona implies one.
"""
from embed_store import get_model, get_collection

# crude persona detection - replace with something better as needed
STAGE_HINTS = {
    "learner": ["learner", "l plate", "l-plate", "learning to drive"],
    "P1": ["p1", "red p", "first year"],
    "P2": ["p2", "green p"],
    "full": ["full licence", "full license"],
}


def detect_stage(question: str) -> str | None:
    lowered = question.lower()
    for stage, hints in STAGE_HINTS.items():
        if any(h in lowered for h in hints):
            return stage
    return None


def retrieve(question: str, top_k: int = 5, filter_stage: bool = True) -> list[dict]:
    """
    Return top_k chunks as list of {text, distance, metadata}.
    If filter_stage=True and a driver stage is detected in the question,
    restrict results to that stage OR stage='all'.
    """
    model = get_model()
    collection = get_collection()
    query_embedding = model.encode([question]).tolist()

    where = None
    if filter_stage:
        stage = detect_stage(question)
        if stage:
            where = {"driver_stage": {"$in": [stage, "all"]}}

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })
    return hits


if __name__ == "__main__":
    q = "Can a P1 driver use a hands-free phone while driving?"
    for hit in retrieve(q):
        print(f"[{hit['distance']:.3f}] {hit['metadata']} -> {hit['text'][:100]}...")
