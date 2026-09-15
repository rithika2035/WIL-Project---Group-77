"""
Stage 5: Generation + guardrail
Builds a citation-forcing prompt from retrieved chunks, calls a local Ollama
model, and falls back to "I don't have enough information" when retrieval
confidence is too low (checked BEFORE calling the LLM, not just via the
prompt instruction - this makes the guardrail testable independent of the
model's own behaviour).
"""
import ollama

OLLAMA_MODEL = "llama3.2:3b"

# Chroma returns cosine *distance* (lower = more similar) when using the
# default embedding space here. Tune this threshold against your own qrels.
DISTANCE_THRESHOLD = 0.6

FALLBACK_MESSAGE = "I don't have enough information to answer that."

PROMPT_TEMPLATE = """You are a road-safety assistant answering questions about \
Victorian (Australia) road rules. Use ONLY the excerpts below to answer.
Cite the source/section for every claim you make, using the format [source: X].
If the excerpts do not contain enough information to confidently answer, \
respond with exactly: "{fallback}"

Excerpts:
{excerpts}

Question: {question}

Answer:"""


def format_excerpts(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        source = h["metadata"].get("source", h["chunk_id"])
        lines.append(f"[source: {source}] {h['text']}")
    return "\n\n".join(lines)


def passes_confidence_check(hits: list[dict], threshold: float = DISTANCE_THRESHOLD) -> bool:
    if not hits:
        return False
    return hits[0]["distance"] <= threshold


def generate_answer(question: str, hits: list[dict]) -> dict:
    """Return {answer, used_fallback, hits}."""
    if not passes_confidence_check(hits):
        return {"answer": FALLBACK_MESSAGE, "used_fallback": True, "hits": hits}

    prompt = PROMPT_TEMPLATE.format(
        fallback=FALLBACK_MESSAGE,
        excerpts=format_excerpts(hits),
        question=question,
    )

    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    answer = response["response"].strip()

    return {"answer": answer, "used_fallback": False, "hits": hits}


if __name__ == "__main__":
    from retrieval import retrieve

    q = "Can a P1 driver use a hands-free phone while driving?"
    hits = retrieve(q)
    result = generate_answer(q, hits)
    print(result["answer"])
