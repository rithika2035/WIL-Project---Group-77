# Technical Report Outline

Fill each section in as the project progresses rather than leaving it all to
Week 6-8 - most of this can be drafted incrementally.

## 1. Problem framing
- The gap: generic road-rule chatbots don't distinguish driver stages, so a
  learner or P1 driver can get an answer that's technically correct for a
  *different* stage.
- Why Victoria only (see docs/business_case.md - frame single-jurisdiction
  scope as enabling depth, not as a limitation).

## 2. Business case
- Paste the finished paragraph from docs/business_case.md here.

## 3. Architecture
- Pipeline diagram: ingestion -> chunking -> embedding -> retrieval -> generation -> guardrail -> UI
- Reference that this reproduces Walert's (RMIT, CHIIR'24) architecture,
  adapted for a different domain and data shape (long-form rules vs FAQ pairs).
- Stack: Ollama (model used), Chroma, sentence-transformers (all-MiniLM-L6-v2).
- Note the two things this project adds beyond Walert's own scope:
  metadata-based driver-stage/topic tagging, and a confidence-threshold
  guardrail checked before generation (not just via prompt instruction).

## 4. Evaluation methodology
- Test set: eval/test_set_template.csv (persona-tagged, includes deliberately
  unanswerable questions to test abstention).
- Metrics: effectiveness (answered/correctly abstained/wrongly abstained/
  wrongly answered), stage/category precision, source attribution
  correctness, faithfulness (manual).
- State how many questions, how many personas, how many were deliberately
  out-of-scope.

## 5. Results
- Paste summary output from `eval/evaluate_full.py` (before/after key design
  changes - e.g. chunk size 500 -> 200 words).
- At least one concrete before/after number backing a design decision
  (checklist explicitly requires this).

## 6. Ethical / social considerations
- Misinformation risk: wrong-stage advice could lead to a fine or unsafe
  behaviour - this is why stage precision is scored separately, not folded
  into general accuracy.
- Data provenance: every chunk traceable to a source URL + access date
  (see data/sources.csv) so answers can be audited.
- Currency of rules: road rules can change; the system has no mechanism to
  detect this automatically (flag as a limitation, see below), so a manual
  re-scrape/re-index cadence should be documented.

## 7. Limitations
- Single jurisdiction (Victoria only) - explicitly frame as scope, not flaw.
- No automated currency/staleness detection.
- Faithfulness metric is manually scored, not automated.
- Keyword-based driver_stage/topic tagger is a placeholder - state whether
  it was replaced and with what.

## 8. Presentation prep
- Live demo script: pick 1 question per persona + 1 deliberate abstention
  example (the "system correctly says it doesn't know" moment is often the
  most persuasive part of a demo).
