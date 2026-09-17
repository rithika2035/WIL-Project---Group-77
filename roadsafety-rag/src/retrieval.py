"""
Stage 4: Retrieval
Embed the incoming question and query Chroma, with optional filtering/
boosting by driver_stage when the question/persona implies one.

IMPORTANT: driver_stage metadata can hold multiple values separated by
semicolons (e.g. "learner;P1;P2" for a passage that applies to several
stages). Chroma's `where` filter only does exact-value matching on scalar
metadata, so stage filtering is done as a POST-FILTER in Python instead:
retrieve a larger candidate pool, then keep passages whose driver_stage
field contains the detected stage (or "all"), then trim to top_k.
"""
CANDIDATE_POOL_MULTIPLIER = 4  # fetch this many x top_k before stage-filtering down

# crude persona/stage detection - replace with something better as needed
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


def stage_matches(metadata_stage: str, target_stage: str) -> bool:
    """metadata_stage is a semicolon-separated string, e.g. 'learner;P1;P2'."""
    values = {v.strip() for v in metadata_stage.split(";") if v.strip()}
    return target_stage in values or "all" in values


def retrieve(question: str, top_k: int = 5, filter_stage: bool = True) -> list[dict]:
    """Return top_k chunks as list of {chunk_id, text, distance, metadata}."""
    from embed_store import get_model, get_collection  # lazy: only needed once chromadb is installed

    model = get_model()
    collection = get_collection()
    query_embedding = model.encode([question]).tolist()

    stage = detect_stage(question) if filter_stage else None
    fetch_n = top_k * CANDIDATE_POOL_MULTIPLIER if stage else top_k

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=fetch_n,
    )

    hits = []
    for i in range(len(results["ids"][0])):
        hits.append({
            "chunk_id": results["ids"][0][i],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    if stage:
        filtered = [h for h in hits if stage_matches(h["metadata"].get("driver_stage", "all"), stage)]
        # fall back to the unfiltered pool if stage-filtering leaves nothing -
        # better to answer with a caveat than to trigger the abstention guardrail
        hits = filtered if filtered else hits

    return hits[:top_k]


if __name__ == "__main__":
    q = "Can a P1 driver use a hands-free phone while driving?"
    for hit in retrieve(q):
        print(f"[{hit['distance']:.3f}] stage={hit['metadata']['driver_stage']} -> {hit['text'][:100]}...")
