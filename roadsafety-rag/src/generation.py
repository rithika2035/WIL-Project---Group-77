"""
Stage 5: Generation + guardrail
Builds a citation-forcing prompt from retrieved chunks, calls a local Ollama
model, and falls back to "I don't have enough information" when retrieval
confidence is too low (checked BEFORE calling the LLM, not just via the
prompt instruction - this makes the guardrail testable independent of the
model's own behaviour).
"""
OLLAMA_MODEL = "llama3.2:3b"

# Chroma returns squared L2 distance by default (not cosine distance), which
# on normalized sentence-transformer embeddings runs roughly 0-4. Genuinely
# relevant top-1 hits in this project have been observed around 0.7-1.2, so
# 0.6 (a cosine-distance-scale guess) was rejecting correct answers.
# 1.1 is a reasonable starting point - but DON'T trust this number blindly.
# Run eval/calibrate_threshold.py against your own labelled questions
# (eval/test_set_from_topics.csv) to pick a value that actually separates
# your "known"/"inferred" questions from your "out_of_kb" ones.
DISTANCE_THRESHOLD = 1.222

FALLBACK_MESSAGE = "I don't have enough information to answer that."

PROMPT_TEMPLATE = """You are a road-safety assistant answering questions about \
Victorian (Australia) road rules. Use ONLY the excerpts below to answer.
Cite the source/section for every claim you make, using the format [source: X].

You must do EXACTLY ONE of the following - never both:
1. If the excerpts contain enough information, give a complete, cited answer
   and STOP. Do not add any disclaimer or hedge sentence after it.
2. If the excerpts do NOT contain enough information, output ONLY this exact
   sentence and nothing else: "{fallback}"

Excerpts:
{excerpts}

Question: {question}

Answer:"""


def format_excerpts(hits: list[dict]) -> str:
    lines = []
    for h in hits:
        meta = h["metadata"]
        # Prefer rich citation (doc, section, page) when available - this is the
        # collection.csv path. Fall back to bare source filename for the
        # raw-PDF ingestion path (ingestion.py/chunking.py), which doesn't
        # know page/section boundaries.
        if meta.get("section_title"):
            citation = f"{meta.get('doc_id', '')} - \"{meta['section_title']}\""
            if meta.get("page_number"):
                citation += f" (p.{meta['page_number']})"
        else:
            citation = meta.get("source", h["chunk_id"])
        lines.append(f"[source: {citation}] {h['text']}")
    return "\n\n".join(lines)


def passes_confidence_check(hits: list[dict], threshold: float = DISTANCE_THRESHOLD) -> bool:
    if not hits:
        return False
    return hits[0]["distance"] <= threshold


def strip_redundant_fallback(answer: str) -> str:
    """Small local models sometimes hedge: they give a real, cited answer and
    then ALSO tack on the exact fallback sentence out of caution, even though
    they just answered. Since we only reach this point after the retrieval
    confidence gate already passed, a trailing fallback sentence at this
    stage is always a contradiction, not a genuine abstention - strip it
    rather than let the response say two contradictory things at once."""
    if FALLBACK_MESSAGE.lower() not in answer.lower():
        return answer

    # if the fallback IS the entire answer (model genuinely couldn't find
    # anything despite the pre-check passing), leave it as-is
    if len(answer.strip()) <= len(FALLBACK_MESSAGE) + 5:
        return answer

    # otherwise strip the redundant sentence wherever it appears
    import re
    cleaned = re.sub(re.escape(FALLBACK_MESSAGE), "", answer, flags=re.IGNORECASE)
    return cleaned.strip()


def generate_answer(question: str, hits: list[dict]) -> dict:
    """Return {answer, used_fallback, hits}."""
    if not passes_confidence_check(hits):
        return {"answer": FALLBACK_MESSAGE, "used_fallback": True, "hits": hits}

    import ollama  # lazy: only needed once the ollama package is installed

    prompt = PROMPT_TEMPLATE.format(
        fallback=FALLBACK_MESSAGE,
        excerpts=format_excerpts(hits),
        question=question,
    )

    response = ollama.generate(model=OLLAMA_MODEL, prompt=prompt)
    answer = strip_redundant_fallback(response["response"].strip())

    return {"answer": answer, "used_fallback": False, "hits": hits}


if __name__ == "__main__":
    from retrieval import retrieve

    q = "Can a P1 driver use a hands-free phone while driving?"
    hits = retrieve(q)
    result = generate_answer(q, hits)
    print(result["answer"])
